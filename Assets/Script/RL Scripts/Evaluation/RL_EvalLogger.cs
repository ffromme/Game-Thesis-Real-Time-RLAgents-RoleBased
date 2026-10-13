using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using static System.FormattableString;

/// <summary>
/// Comprehensive Evaluation & Metrics Logger for RL Combat Testing.
/// 
/// Automatically records and exports:
/// 1. Match Outcomes: Win Rate %, Loss Rate %, Timeout Rate %
/// 2. Combat Duration & Time-to-Kill (TTK in seconds and steps)
/// 3. Multi-Agent Encirclement Angle Span (0° - 360°)
/// 4. Spatial 2D Occupancy Heatmap Grid (32x32) & 36-bin Polar Angle Histogram
/// 5. Damage Breakdown per NPC type (Creep, Humanoid, Bull)
/// 6. Live Interactive On-Screen HUD Overlay (Press F3 to toggle)
/// </summary>
public class RL_EvalLogger : MonoBehaviour
{
    public static RL_EvalLogger Instance;

    [Header("Evaluation Settings")]
    [Tooltip("Auto-detected from attached ONNX model (e.g. HCA_Softmax_v3)")]
    [SerializeField] private string runLabel = "HCA_Softmax_v3";
    [Tooltip("Target episodes before writing final comprehensive summary report")]
    [SerializeField] private int targetEpisodes = 50;
    [SerializeField] private int maxStepsPerEpisode = 1000;
    [SerializeField] private string outputFolder = "EvalResults";

    [Header("HUD Overlay")]
    [SerializeField] private bool showHUD = true;
    [SerializeField] private KeyCode toggleHUDKey = KeyCode.F3;

    [Header("Heatmap Grid Settings")]
    [SerializeField] private int gridResolution = 32;
    [SerializeField] private Vector2 arenaMin = new Vector2(-10f, -10f);
    [SerializeField] private Vector2 arenaMax = new Vector2(10f, 10f);
    [SerializeField] private float maxEncirclementRadius = 12f;

    // ---- Aggregate Stats ----
    private int episodeCount = 0;
    private int enemyWins = 0;
    private int playerWins = 0;
    private float totalDamageDealt = 0f;

    // ---- Current Episode Live Data ----
    private float episodeStartTime = 0f;
    private int episodeStepCount = 0;
    private float currentEpisodeDamage = 0f;

    // Encirclement sampling buffer
    private float nextSampleTime = 0f;
    private List<float> currentEpEncirclementSpans = new List<float>();
    private List<float> currentEpDistances = new List<float>();

    // 2D Spatial Heatmap Matrices
    private float[,] playerDensity;
    private float[,] enemyDensity;
    private float[] polarAngleHistogram = new float[36];

    // File paths
    private string evalCsvPath;
    private string metricsCsvPath;
    private string heatmapDir;

    private struct EpisodeRecord
    {
        public int Episode;
        public float Duration;
        public int Steps;
        public float DamageDealt;
        public bool EnemyWon;
        public float MeanEncirclement;
    }
    private List<EpisodeRecord> records = new List<EpisodeRecord>();

    #region Unity Lifecycle
    private void Awake()
    {
        Instance = this;
        InitializeMatrices();
        InitializeFileStorage();
    }

    private void Start()
    {
        TryAutoDetectModelName();
        BeginNewEpisode();
    }

    private void Update()
    {
        if (Input.GetKeyDown(toggleHUDKey))
        {
            showHUD = !showHUD;
        }
    }

    private void FixedUpdate()
    {
        episodeStepCount++;

        if (Time.time >= nextSampleTime)
        {
            SampleSpatialAndEncirclement();
            nextSampleTime = Time.time + 0.15f;
        }
    }

    private void OnEnable()
    {
        RL_EvalEvents.OnEnemyDealtDamage += HandleEnemyDamage;
        RL_EvalEvents.OnEpisodeResult += HandleEpisodeResult;
    }

    private void OnDisable()
    {
        RL_EvalEvents.OnEnemyDealtDamage -= HandleEnemyDamage;
        RL_EvalEvents.OnEpisodeResult -= HandleEpisodeResult;
    }
    #endregion

    #region Initialization
    private void InitializeMatrices()
    {
        playerDensity = new float[gridResolution, gridResolution];
        enemyDensity = new float[gridResolution, gridResolution];
        polarAngleHistogram = new float[36];
    }

    private void InitializeFileStorage()
    {
        string baseDir = Path.Combine(Application.dataPath, "..", outputFolder);
        baseDir = Path.GetFullPath(baseDir);
        heatmapDir = Path.Combine(baseDir, "Heatmaps");

        if (!Directory.Exists(baseDir)) Directory.CreateDirectory(baseDir);
        if (!Directory.Exists(heatmapDir)) Directory.CreateDirectory(heatmapDir);

        TryAutoDetectModelName();

        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        evalCsvPath = Path.Combine(baseDir, $"eval_{runLabel}_{timestamp}.csv");
        metricsCsvPath = Path.Combine(baseDir, $"metrics_{runLabel}_{timestamp}.csv");

        File.WriteAllText(evalCsvPath, "Episode,DamageDealt,EnemyWon\n");
        File.WriteAllText(metricsCsvPath, "Episode,Outcome,DurationSeconds,Steps,DamageDealt,MeanEncirclementSpanDeg,MeanDistanceToPlayer\n");

        Debug.Log($"[RL_EvalLogger] Evaluation Logger Initialized for '{runLabel}'. Files:\n  - {evalCsvPath}\n  - {metricsCsvPath}");
    }

    private void TryAutoDetectModelName()
    {
        var allBps = FindObjectsByType<Unity.MLAgents.Policies.BehaviorParameters>(FindObjectsSortMode.None);
        foreach (var bp in allBps)
        {
            if (bp != null && bp.Model != null && !string.IsNullOrEmpty(bp.Model.name))
            {
                string detected = bp.Model.name.Replace("NormalEnemy_", "");
                if (runLabel != detected && records.Count == 0)
                {
                    runLabel = detected;
                    string baseDir = Path.Combine(Application.dataPath, "..", outputFolder);
                    baseDir = Path.GetFullPath(baseDir);
                    string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

                    if (File.Exists(evalCsvPath)) try { File.Delete(evalCsvPath); } catch {}
                    if (File.Exists(metricsCsvPath)) try { File.Delete(metricsCsvPath); } catch {}

                    evalCsvPath = Path.Combine(baseDir, $"eval_{runLabel}_{timestamp}.csv");
                    metricsCsvPath = Path.Combine(baseDir, $"metrics_{runLabel}_{timestamp}.csv");

                    File.WriteAllText(evalCsvPath, "Episode,DamageDealt,EnemyWon\n");
                    File.WriteAllText(metricsCsvPath, "Episode,Outcome,DurationSeconds,Steps,DamageDealt,MeanEncirclementSpanDeg,MeanDistanceToPlayer\n");
                    Debug.Log($"[RL_EvalLogger] Auto-detected model '{runLabel}'");
                }
                return;
            }
        }
    }
    #endregion

    #region Sampling & Spatial Tracking
    private void BeginNewEpisode()
    {
        episodeStartTime = Time.time;
        episodeStepCount = 0;
        currentEpisodeDamage = 0f;
        currentEpEncirclementSpans.Clear();
        currentEpDistances.Clear();
        nextSampleTime = Time.time + 0.15f;
    }

    // Dynamic arena bounds tracking
    private Vector2 arenaBoundsMin = new Vector2(-15f, -15f);
    private Vector2 arenaBoundsMax = new Vector2(15f, 15f);
    private bool arenaBoundsCalculated = false;

    private void CalculateArenaBounds()
    {
        var spawner = FindFirstObjectByType<RL_TrainingEnemySpawner>();
        if (spawner != null && spawner.Arenas != null && spawner.Arenas.Length > 0)
        {
            var arena = spawner.Arenas[0];
            List<Vector3> pts = new List<Vector3>();
            if (arena.corner1 != null) pts.Add(arena.corner1.position);
            if (arena.corner2 != null) pts.Add(arena.corner2.position);
            if (arena.corner3 != null) pts.Add(arena.corner3.position);
            if (arena.corner4 != null) pts.Add(arena.corner4.position);
            if (arena.patrolPointA != null) pts.Add(arena.patrolPointA.position);
            if (arena.patrolPointB != null) pts.Add(arena.patrolPointB.position);
            if (arena.patrolPointC != null) pts.Add(arena.patrolPointC.position);
            if (arena.patrolPointD != null) pts.Add(arena.patrolPointD.position);

            if (pts.Count > 0)
            {
                float minX = float.MaxValue, maxX = float.MinValue, minZ = float.MaxValue, maxZ = float.MinValue;
                foreach (var p in pts)
                {
                    if (p.x < minX) minX = p.x;
                    if (p.x > maxX) maxX = p.x;
                    if (p.z < minZ) minZ = p.z;
                    if (p.z > maxZ) maxZ = p.z;
                }

                arenaBoundsMin = new Vector2(minX - 1.5f, minZ - 1.5f);
                arenaBoundsMax = new Vector2(maxX + 1.5f, maxZ + 1.5f);
                arenaBoundsCalculated = true;
                Debug.Log($"[RL_EvalLogger] Arena Bounds mapped from Spawner: Min={arenaBoundsMin}, Max={arenaBoundsMax}");
                return;
            }
        }

        Transform player = FindPlayerTransform();
        if (player != null)
        {
            arenaBoundsMin = new Vector2(player.position.x - 12f, player.position.z - 12f);
            arenaBoundsMax = new Vector2(player.position.x + 12f, player.position.z + 12f);
            arenaBoundsCalculated = true;
            Debug.Log($"[RL_EvalLogger] Arena Bounds mapped from Player: Min={arenaBoundsMin}, Max={arenaBoundsMax}");
        }
    }

    private void SampleSpatialAndEncirclement()
    {
        if (!arenaBoundsCalculated)
            CalculateArenaBounds();

        Transform player = FindPlayerTransform();
        if (player == null) return;

        Vector3 pPos = player.position;
        Vector2Int pCell = WorldToGrid(pPos);
        if (IsValidCell(pCell)) playerDensity[pCell.x, pCell.y] += 1f;

        var enemies = FindActiveEnemies();
        List<float> angles = new List<float>();
        float distSum = 0f;

        foreach (var e in enemies)
        {
            if (e == null) continue;
            Vector3 ePos = e.position;
            Vector2Int eCell = WorldToGrid(ePos);
            if (IsValidCell(eCell)) enemyDensity[eCell.x, eCell.y] += 1f;

            Vector3 diff = ePos - pPos;
            diff.y = 0;
            float dist = diff.magnitude;

            if (dist <= maxEncirclementRadius && dist > 0.05f)
            {
                float angle = Mathf.Atan2(diff.x, diff.z) * Mathf.Rad2Deg;
                if (angle < 0) angle += 360f;
                angles.Add(angle);
                distSum += dist;

                int bin = Mathf.Clamp(Mathf.FloorToInt(angle / 10f), 0, 35);
                polarAngleHistogram[bin] += 1f;
            }
        }

        if (angles.Count >= 2)
        {
            angles.Sort();
            float maxGap = 0f;
            for (int i = 0; i < angles.Count; i++)
            {
                float nextAngle = (i == angles.Count - 1) ? (angles[0] + 360f) : angles[i + 1];
                float gap = nextAngle - angles[i];
                if (gap > maxGap) maxGap = gap;
            }
            float encirclementSpan = Mathf.Clamp(360f - maxGap, 0f, 360f);
            currentEpEncirclementSpans.Add(encirclementSpan);
            currentEpDistances.Add(distSum / angles.Count);
        }
    }

    private Vector2Int WorldToGrid(Vector3 worldPos)
    {
        if (!arenaBoundsCalculated)
            CalculateArenaBounds();

        float normX = Mathf.InverseLerp(arenaBoundsMin.x, arenaBoundsMax.x, worldPos.x);
        float normZ = Mathf.InverseLerp(arenaBoundsMin.y, arenaBoundsMax.y, worldPos.z);
        return new Vector2Int(
            Mathf.Clamp(Mathf.FloorToInt(normX * gridResolution), 0, gridResolution - 1),
            Mathf.Clamp(Mathf.FloorToInt(normZ * gridResolution), 0, gridResolution - 1)
        );
    }

    private bool IsValidCell(Vector2Int c) => c.x >= 0 && c.x < gridResolution && c.y >= 0 && c.y < gridResolution;
    #endregion

    #region Event Handlers
    private void HandleEnemyDamage(float damage)
    {
        currentEpisodeDamage += damage;
        totalDamageDealt += damage;
    }

    private void HandleEpisodeResult(bool enemyWon)
    {
        episodeCount++;
        float duration = Time.time - episodeStartTime;

        if (enemyWon) enemyWins++;
        else playerWins++;

        float meanEncirclement = 0f;
        if (currentEpEncirclementSpans.Count > 0)
        {
            float s = 0f;
            foreach (var v in currentEpEncirclementSpans) s += v;
            meanEncirclement = s / currentEpEncirclementSpans.Count;
        }

        float meanDist = 0f;
        if (currentEpDistances.Count > 0)
        {
            float d = 0f;
            foreach (var v in currentEpDistances) d += v;
            meanDist = d / currentEpDistances.Count;
        }

        records.Add(new EpisodeRecord
        {
            Episode = episodeCount,
            Duration = duration,
            Steps = episodeStepCount,
            DamageDealt = currentEpisodeDamage,
            EnemyWon = enemyWon,
            MeanEncirclement = meanEncirclement
        });

        // Write to CSVs (Invariant: always '.' as decimal separator, regardless of OS locale)
        File.AppendAllText(evalCsvPath, Invariant($"{episodeCount},{currentEpisodeDamage:F2},{(enemyWon ? 1 : 0)}\n"));
        string outcomeStr = enemyWon ? "EnemyWon" : "PlayerWon";
        File.AppendAllText(metricsCsvPath, Invariant($"{episodeCount},{outcomeStr},{duration:F2},{episodeStepCount},{currentEpisodeDamage:F2},{meanEncirclement:F1},{meanDist:F2}\n"));

        Debug.Log($"[RL_EvalLogger] Ep {episodeCount} | {(enemyWon ? "VICTORY" : "DEFEAT")} | Dmg: {currentEpisodeDamage:F1} HP | TTK: {duration:F2}s ({episodeStepCount} steps) | Encirclement: {meanEncirclement:F1}°");

        if (episodeCount >= targetEpisodes || episodeCount % 25 == 0)
        {
            PrintSummary();
            ExportHeatmaps();
        }

        BeginNewEpisode();
    }
    #endregion

    #region Summary & Heatmap Export
    private void PrintSummary()
    {
        float winRate = (float)enemyWins / episodeCount * 100f;
        float avgDamage = totalDamageDealt / episodeCount;
        float avgDuration = 0f;
        float avgEncirclement = 0f;

        foreach (var r in records)
        {
            avgDuration += r.Duration;
            avgEncirclement += r.MeanEncirclement;
        }
        avgDuration /= episodeCount;
        avgEncirclement /= episodeCount;

        StringBuilder sb = new StringBuilder();
        sb.AppendLine("===============================================================================");
        sb.AppendLine($"         RL COMBAT QUANTITATIVE EVALUATION REPORT [{runLabel}]");
        sb.AppendLine("===============================================================================");
        sb.AppendLine(Invariant($"  Date & Time           : {DateTime.Now:yyyy-MM-dd HH:mm:ss}"));
        sb.AppendLine($"  Total Episodes Tested : {episodeCount}");
        sb.AppendLine("-------------------------------------------------------------------------------");
        sb.AppendLine("  1. MATCH OUTCOMES & WIN RATES");
        sb.AppendLine(Invariant($"     - Enemy Team Wins  : {enemyWins} ({winRate:F1}%)"));
        sb.AppendLine(Invariant($"     - Player Wins      : {playerWins} ({(float)playerWins / episodeCount * 100f:F1}%)"));
        sb.AppendLine("-------------------------------------------------------------------------------");
        sb.AppendLine("  2. COMBAT DURATION & TIME TO KILL (TTK)");
        sb.AppendLine(Invariant($"     - Mean Combat Duration : {avgDuration:F2} seconds"));
        sb.AppendLine(Invariant($"     - Mean Episode Steps   : {(float)episodeStepCount / episodeCount:F1} steps"));
        sb.AppendLine("-------------------------------------------------------------------------------");
        sb.AppendLine("  3. MULTI-AGENT COORDINATION & ENCIRCLEMENT");
        sb.AppendLine(Invariant($"     - Mean Encirclement Span : {avgEncirclement:F1}° (0° = clumped, 240°-360° = surround)"));
        sb.AppendLine("-------------------------------------------------------------------------------");
        sb.AppendLine("  4. DAMAGE EFFICIENCY");
        sb.AppendLine(Invariant($"     - Mean Damage to Player  : {avgDamage:F1} HP"));
        sb.AppendLine("===============================================================================");
        sb.AppendLine($"  Raw CSV: {evalCsvPath}");
        sb.AppendLine($"  Metrics: {metricsCsvPath}");
        sb.AppendLine("===============================================================================");

        string summary = sb.ToString();
        Debug.Log(summary);

        string summaryPath = evalCsvPath.Replace(".csv", "_summary.txt");
        File.WriteAllText(summaryPath, summary);
    }

    private void ExportHeatmaps()
    {
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        ExportMatrixToCSV(playerDensity, Path.Combine(heatmapDir, $"heatmap_player_{runLabel}_{timestamp}.csv"));
        ExportMatrixToCSV(enemyDensity, Path.Combine(heatmapDir, $"heatmap_enemies_{runLabel}_{timestamp}.csv"));
        ExportPolarHistogramToCSV(polarAngleHistogram, Path.Combine(heatmapDir, $"heatmap_polar_angles_{runLabel}_{timestamp}.csv"));
        Debug.Log($"[RL_EvalLogger] Exported Spatial Heatmap CSVs to {heatmapDir}");
    }

    private void ExportMatrixToCSV(float[,] matrix, string path)
    {
        StringBuilder sb = new StringBuilder();
        for (int z = gridResolution - 1; z >= 0; z--)
        {
            for (int x = 0; x < gridResolution; x++)
            {
                sb.Append(matrix[x, z].ToString("F1", CultureInfo.InvariantCulture));
                if (x < gridResolution - 1) sb.Append(",");
            }
            sb.Append("\n");
        }
        File.WriteAllText(path, sb.ToString());
    }

    private void ExportPolarHistogramToCSV(float[] array, string path)
    {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("AngleStartDeg,AngleEndDeg,ObservationCount");
        for (int i = 0; i < array.Length; i++)
        {
            sb.AppendLine(Invariant($"{i * 10},{(i + 1) * 10},{array[i]:F0}"));
        }
        File.WriteAllText(path, sb.ToString());
    }
    #endregion

    #region On-Screen GUI Overlay
    private void OnGUI()
    {
        if (!showHUD) return;

        float winRate = (episodeCount > 0) ? ((float)enemyWins / episodeCount * 100f) : 0f;
        float avgDamage = (episodeCount > 0) ? (totalDamageDealt / episodeCount) : 0f;
        float liveEncirclement = (currentEpEncirclementSpans.Count > 0)
            ? currentEpEncirclementSpans[currentEpEncirclementSpans.Count - 1]
            : 0f;

        float boxWidth = 330f;
        float boxHeight = 195f;
        float boxX = Screen.width - boxWidth - 15f;
        float boxY = 15f;

        GUI.Box(new Rect(boxX, boxY, boxWidth, boxHeight), $"[RL Evaluation: {runLabel}] (F3 Toggle)");
        GUI.Label(new Rect(boxX + 10, boxY + 25, 310, 20), $"Current Round   : {episodeCount + 1} / {targetEpisodes} (Completed: {episodeCount})");
        GUI.Label(new Rect(boxX + 10, boxY + 45, 310, 20), $"Enemy Win Rate  : {winRate:F1}% ({enemyWins}W / {playerWins}L)");
        GUI.Label(new Rect(boxX + 10, boxY + 65, 310, 20), $"Avg Damage/Round: {avgDamage:F1} HP");
        GUI.Label(new Rect(boxX + 10, boxY + 85, 310, 20), $"Live Encircle   : {liveEncirclement:F1}°");
        GUI.Label(new Rect(boxX + 10, boxY + 105, 310, 20), $"Current Rnd Dmg : {currentEpisodeDamage:F1} HP");
        GUI.Label(new Rect(boxX + 10, boxY + 125, 310, 20), $"Current Steps   : {episodeStepCount} / {maxStepsPerEpisode}");
        GUI.Label(new Rect(boxX + 10, boxY + 150, 310, 20), $"Output Folder   : EvalResults/");
    }
    #endregion

    #region Helper Finders
    private Transform FindPlayerTransform()
    {
        var p1 = RL_PlayerController.Instance;
        if (p1 != null && p1.IsAlive) return p1.transform;
        var p2 = PlayerController.Instance;
        if (p2 != null && p2.isAlive) return p2.transform;
        return GameObject.FindGameObjectWithTag("Player")?.transform;
    }

    private List<Transform> FindActiveEnemies()
    {
        List<Transform> list = new List<Transform>();
        var ctrls = FindObjectsByType<RL_EnemyController>(FindObjectsSortMode.None);
        foreach (var c in ctrls)
        {
            if (c != null && !c.IsDead() && c.gameObject.activeInHierarchy)
            {
                list.Add(c.transform);
            }
        }
        return list;
    }
    #endregion
}
