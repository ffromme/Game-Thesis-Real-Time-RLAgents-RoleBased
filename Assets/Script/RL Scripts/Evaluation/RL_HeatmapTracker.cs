using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using static System.FormattableString;

/// <summary>
/// Spatial 2D Heatmap & Encirclement Trajectory Tracker.
/// 
/// Records and visualizes spatial movement patterns for thesis evaluation:
/// 1. Player & NPC Spatial Occupancy Matrices (Grid 32x32)
/// 2. Encirclement Polar Angle Histogram (36 bins: 0° - 360°)
/// 3. In-Editor Live Floor Gizmos Heatmap (Cold Blue -> Warm Yellow -> Hot Red)
/// 4. Automatic CSV Matrix Export for publication-ready Python plotting.
/// </summary>
public class RL_HeatmapTracker : MonoBehaviour
{
    public static RL_HeatmapTracker Instance;

    [Header("Evaluation Settings")]
    [Tooltip("Label for the evaluation run (e.g. PPO_Baseline, HCA_Thesis_Run1)")]
    [SerializeField] private string runLabel = "HCA_Thesis_Run1";
    [Tooltip("Sample spatial positions every N seconds")]
    [SerializeField] private float sampleInterval = 0.1f;
    [Tooltip("Export CSV files after this many episodes")]
    [SerializeField] private int autoExportAfterEpisodes = 30;
    [SerializeField] private string outputDirectory = "EvalResults/Heatmaps";

    [Header("Arena Bounds (Grid Discretization)")]
    [SerializeField] private int gridResolution = 32;
    [SerializeField] private Vector2 arenaMin = new Vector2(-10f, -10f);
    [SerializeField] private Vector2 arenaMax = new Vector2(10f, 10f);
    [SerializeField] private float arenaFloorY = 0.85f;

    [Header("In-Editor Gizmo Visualization")]
    [SerializeField] private bool drawGizmosHeatmap = true;
    [SerializeField] private bool showPlayerHeatmap = true;
    [SerializeField] private bool showEnemyHeatmap = true;
    [SerializeField] [Range(0.05f, 1f)] private float gizmoAlpha = 0.45f;

    // ---- 2D Grid Accumulators ----
    private float[,] playerDensity;
    private float[,] enemyDensity;
    private float[,] creepDensity;
    private float[,] humanoidDensity;
    private float[,] bullDensity;

    // ---- Polar Angle Histogram (36 bins = 10 deg each) ----
    private float[] polarAngleHistogram = new float[36];

    private float nextSampleTime = 0f;
    private int completedEpisodes = 0;
    private float maxRecordedDensity = 1f;

    #region Unity Lifecycle
    private void Awake()
    {
        Instance = this;
        InitializeGridMatrices();
    }

    private void OnEnable()
    {
        RL_EvalEvents.OnEpisodeResult += OnEpisodeFinished;
    }

    private void OnDisable()
    {
        RL_EvalEvents.OnEpisodeResult -= OnEpisodeFinished;
    }

    private void FixedUpdate()
    {
        if (Time.time >= nextSampleTime)
        {
            RecordSpatialSample();
            nextSampleTime = Time.time + sampleInterval;
        }
    }
    #endregion

    #region Initialization
    private void InitializeGridMatrices()
    {
        playerDensity = new float[gridResolution, gridResolution];
        enemyDensity = new float[gridResolution, gridResolution];
        creepDensity = new float[gridResolution, gridResolution];
        humanoidDensity = new float[gridResolution, gridResolution];
        bullDensity = new float[gridResolution, gridResolution];
        polarAngleHistogram = new float[36];
    }
    #endregion

    #region Spatial Sampling Engine
    private void RecordSpatialSample()
    {
        Transform player = FindPlayer();
        Vector3 playerPos = (player != null) ? player.position : Vector3.zero;

        // 1. Record Player Position
        if (player != null)
        {
            Vector2Int pCell = WorldToGrid(playerPos.x, playerPos.z);
            if (IsValidCell(pCell))
            {
                playerDensity[pCell.x, pCell.y] += 1f;
                if (playerDensity[pCell.x, pCell.y] > maxRecordedDensity)
                    maxRecordedDensity = playerDensity[pCell.x, pCell.y];
            }
        }

        // 2. Record Enemy Positions & Relative Polar Angles
        var enemies = FindActiveEnemies();
        foreach (var enemy in enemies)
        {
            if (enemy == null) continue;
            Vector3 ePos = enemy.position;
            Vector2Int eCell = WorldToGrid(ePos.x, ePos.z);

            if (IsValidCell(eCell))
            {
                enemyDensity[eCell.x, eCell.y] += 1f;
                if (enemyDensity[eCell.x, eCell.y] > maxRecordedDensity)
                    maxRecordedDensity = enemyDensity[eCell.x, eCell.y];

                var ctrl = enemy.GetComponent<RL_EnemyController>();
                if (ctrl != null)
                {
                    switch (ctrl.enemyType)
                    {
                        case EnemyType.Creep:
                            creepDensity[eCell.x, eCell.y] += 1f;
                            break;
                        case EnemyType.Medium1:
                            humanoidDensity[eCell.x, eCell.y] += 1f;
                            break;
                        case EnemyType.Medium2:
                            bullDensity[eCell.x, eCell.y] += 1f;
                            break;
                    }
                }
            }

            // 3. Record Polar Angle relative to player
            if (player != null)
            {
                Vector3 diff = ePos - playerPos;
                diff.y = 0;
                if (diff.sqrMagnitude > 0.01f)
                {
                    float angle = Mathf.Atan2(diff.x, diff.z) * Mathf.Rad2Deg;
                    if (angle < 0) angle += 360f;
                    int bin = Mathf.Clamp(Mathf.FloorToInt(angle / 10f), 0, 35);
                    polarAngleHistogram[bin] += 1f;
                }
            }
        }
    }

    private Vector2Int WorldToGrid(float worldX, float worldZ)
    {
        float normX = Mathf.InverseLerp(arenaMin.x, arenaMax.x, worldX);
        float normZ = Mathf.InverseLerp(arenaMin.y, arenaMax.y, worldZ);

        int gx = Mathf.Clamp(Mathf.FloorToInt(normX * gridResolution), 0, gridResolution - 1);
        int gz = Mathf.Clamp(Mathf.FloorToInt(normZ * gridResolution), 0, gridResolution - 1);
        return new Vector2Int(gx, gz);
    }

    private Vector3 GridToWorld(int gx, int gz)
    {
        float wx = Mathf.Lerp(arenaMin.x, arenaMax.x, (gx + 0.5f) / gridResolution);
        float wz = Mathf.Lerp(arenaMin.y, arenaMax.y, (gz + 0.5f) / gridResolution);
        return new Vector3(wx, arenaFloorY, wz);
    }

    private bool IsValidCell(Vector2Int c) => c.x >= 0 && c.x < gridResolution && c.y >= 0 && c.y < gridResolution;

    private void OnEpisodeFinished(bool enemyWon)
    {
        completedEpisodes++;
        if (completedEpisodes >= autoExportAfterEpisodes)
        {
            ExportAllHeatmapsToCSV();
        }
    }
    #endregion

    #region CSV Export System
    public void ExportAllHeatmapsToCSV()
    {
        TryAutoDetectModelName();

        string fullDirPath = Path.Combine(Application.dataPath, "..", outputDirectory);
        fullDirPath = Path.GetFullPath(fullDirPath);

        if (!Directory.Exists(fullDirPath))
        {
            Directory.CreateDirectory(fullDirPath);
        }

        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

        ExportMatrixToCSV(playerDensity, Path.Combine(fullDirPath, $"heatmap_player_{runLabel}_{timestamp}.csv"));
        ExportMatrixToCSV(enemyDensity, Path.Combine(fullDirPath, $"heatmap_enemies_{runLabel}_{timestamp}.csv"));
        ExportMatrixToCSV(creepDensity, Path.Combine(fullDirPath, $"heatmap_creep_{runLabel}_{timestamp}.csv"));
        ExportMatrixToCSV(humanoidDensity, Path.Combine(fullDirPath, $"heatmap_humanoid_{runLabel}_{timestamp}.csv"));
        ExportMatrixToCSV(bullDensity, Path.Combine(fullDirPath, $"heatmap_bull_{runLabel}_{timestamp}.csv"));
        ExportArrayToCSV(polarAngleHistogram, Path.Combine(fullDirPath, $"heatmap_polar_angles_{runLabel}_{timestamp}.csv"));

        Debug.Log($"[RL_HeatmapTracker] Successfully exported 6 heatmap CSVs for '{runLabel}' to {fullDirPath}");
    }

    private void TryAutoDetectModelName()
    {
        var allBps = FindObjectsByType<Unity.MLAgents.Policies.BehaviorParameters>(FindObjectsSortMode.None);
        foreach (var bp in allBps)
        {
            if (bp != null && bp.Model != null && !string.IsNullOrEmpty(bp.Model.name))
            {
                runLabel = bp.Model.name.Replace("NormalEnemy_", "");
                return;
            }
        }
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

    private void ExportArrayToCSV(float[] array, string path)
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

    #region Helper Finders
    private Transform FindPlayer()
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

    #region In-Editor Gizmos Heatmap Rendering
    private void OnDrawGizmosSelected()
    {
        if (!drawGizmosHeatmap || playerDensity == null) return;

        float cellSizeX = (arenaMax.x - arenaMin.x) / gridResolution;
        float cellSizeZ = (arenaMax.y - arenaMin.y) / gridResolution;
        Vector3 size = new Vector3(cellSizeX * 0.95f, 0.05f, cellSizeZ * 0.95f);

        for (int x = 0; x < gridResolution; x++)
        {
            for (int z = 0; z < gridResolution; z++)
            {
                float val = 0f;
                if (showPlayerHeatmap) val += playerDensity[x, z];
                if (showEnemyHeatmap) val += enemyDensity[x, z];

                if (val > 0.1f)
                {
                    float heat = Mathf.Clamp01(val / (maxRecordedDensity * 0.7f));
                    Gizmos.color = GetHeatColor(heat, gizmoAlpha);
                    Vector3 center = GridToWorld(x, z);
                    Gizmos.DrawCube(center, size);
                }
            }
        }
    }

    private Color GetHeatColor(float normalizedVal, float alpha)
    {
        // Cold Blue (0.0) -> Cyan (0.25) -> Green (0.5) -> Yellow (0.75) -> Hot Red (1.0)
        Color c;
        if (normalizedVal < 0.25f)
            c = Color.Lerp(Color.blue, Color.cyan, normalizedVal / 0.25f);
        else if (normalizedVal < 0.5f)
            c = Color.Lerp(Color.cyan, Color.green, (normalizedVal - 0.25f) / 0.25f);
        else if (normalizedVal < 0.75f)
            c = Color.Lerp(Color.green, Color.yellow, (normalizedVal - 0.5f) / 0.25f);
        else
            c = Color.Lerp(Color.yellow, Color.red, (normalizedVal - 0.75f) / 0.25f);

        c.a = alpha;
        return c;
    }
    #endregion
}
