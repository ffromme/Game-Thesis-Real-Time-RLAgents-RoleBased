# AGENTS.md — code/ (Code Crusader, jalur code skripsi Iam)

Bagian atas file ini ditulis untuk skripsi Iam (Kandidat F-α). Bagian bawah,
"Catatan teknis repo (dari kating)", adalah AGENTS.md asli repo milik kating
dan dipertahankan apa adanya.

Konteks riset (fokus F-α, keputusan, larangan umum) ada di `../AGENTS.md`.
Claude Code membacanya otomatis bila dijalankan dari folder ini. Alat yang
hanya membaca sampai root git (mis. codex) harus membaca `../AGENTS.md` dan
`../handoff/README.md` secara manual di awal sesi.

## Asal repo
- Repo riset kating (Rava Raditya Razan): NPC musuh Code Crusader, HCA
  (hierarchical critic) vs PPO, Unity ML-Agents. Nama folder asli:
  `Game-Thesis-Real-Time-RLAgents-5.0`.
- Kondisi kode kating = baseline B0. Pekerjaan skripsi dibuat di branch
  sendiri (mis. `iam/f-alpha`); buat tag `kating-baseline` sebelum perubahan
  pertama supaya baseline selalu bisa dikembalikan.

## Stack (dibaca dari file repo, belum diverifikasi dengan menjalankan)
- Unity Editor 6000.0.38f1; package `com.unity.ml-agents` 4.0.2.
- Python 3.10.12 (conda env `mlagents`, lihat `environment.yml`);
  `mlagents`/`mlagents-envs` 1.1.0; torch 2.2.2+cu121.
- Trainer HCA: vendored di `ml-agents/ml-agents/mlagents/trainers/hca/`.
- Config: `config/ppo/`, `config/hca/` (varian Softmax/Max, 50M step).
- Hasil training: `results/<run-id>/`; evaluasi gameplay: `EvalResults/`
  (dibuat `RL_EvalLogger`); skrip analisis: `scripts/`.

## Hal yang perlu disesuaikan di laptop Iam
- Skrip `run_all_experiments*.ps1`, `resume_hca_softmax_50M.ps1`, dan
  `MLAgents COnda Script.txt` memakai path milik kating
  (`C:\Users\RavaRazan\...`). Ganti ke path lokal sebelum dipakai; jangan
  menimpa versi kating, buat salinan/parameter.
- `.gitignore` berisi sisa konflik merge (`<<<<<<< Updated upstream` …
  `>>>>>>> Stashed changes`). Rapikan dulu supaya `*.pt`, `*.log`, dan
  `run_logs/` benar-benar diabaikan.
- Kating memakai 8 env paralel (`--num-envs=8`, `--no-graphics`, build
  `.exe`). Laptop Iam: RTX 3050 6 GB, RAM 16 GB → ukur dulu berapa env yang
  muat sebelum run panjang.
- Config tidak menetapkan seed; tetapkan lewat `--seed=<n>` di CLI agar run
  bisa diulang.

## Tugas agent di sini
1. Ambil tugas dari `../handoff/to-code/`. Jangan mengarang eksperimen
   sendiri; kalau spesifikasi ambigu, tanya Iam.
2. Implementasi bertahap: tunjukkan langkah demi langkah supaya hasil tiap
   langkah bisa dievaluasi Iam sebelum lanjut.
3. Setelah run selesai, tulis laporan di
   `../handoff/from-code/YYYY-MM-DD-<run-id>.md` (pakai
   `_template-run-report.md`).
4. Kalau arsitektur, observasi, reward, atau perintah berubah, perbarui
   `../handoff/from-code/code-summary.md`.

## Aturan eksperimen
- Run-id: `<varian>-s<seed>-<YYYYMMDD>` (mis. `B0-s1-20261010`). Jangan
  memakai ulang run-id; jangan `--force` ke run-id milik kating.
- Catat untuk setiap run: commit hash, file config, seed, max_steps, jumlah
  env, mesin (laptop/PC lab), waktu wall-clock, throughput (step/detik),
  lokasi `results/<run-id>` dan file `EvalResults/`.
- Pilot pendek dulu sebelum run panjang.
- Jangan menghapus atau menimpa `results/`, `EvalResults/`, atau model
  `.onnx` kating tanpa izin Iam.
- Perubahan observasi/aksi wajib sinkron di C#, prefab, dan YAML (lihat
  checklist di bawah).

## Penjelasan ke Iam
Iam masih awam di NN/RL. Jelaskan perubahan kode dan hasil training dengan
bahasa sederhana; rujuk `../research/docs/00-context/kamus-istilah-f.md`.

---

## Catatan teknis repo (dari kating, asli)

### Scope and priorities
- This repo is a Unity game project with an RL training stack for enemy NPCs; prioritize changes in `Assets/Script/RL Scripts/` and `config/` before touching gameplay scripts.
- The training behavior name is `NormalEnemy`; keep Unity `BehaviorParameters` and YAML behavior keys aligned (see `Assets/Level/RL Scenes/Training/Art/Models/RL Agents/RL_Humanoid.prefab`, `config/ppo/NormalEnemyCC.yaml`, `config/hca/NormalEnemyHCA.yaml`).
- Prefer the active RL scripts under `Assets/Script/RL Scripts/**`; treat top-level backups like `Assets/Script/NormalEnemyAgents Before Update.cs` and `Assets/Script/RL_EnemyController Before Update.cs` as historical snapshots.
- Active agent implementation is now in `Assets/Script/RL Scripts/Normal Enemy/NormalEnemyAgents.cs` (not the root level backup).

### Architecture map (what talks to what)
- `NormalEnemyAgent` (`Assets/Script/RL Scripts/Normal Enemy/NormalEnemyAgents.cs`) is the ML-Agents entrypoint: observations, action processing, rewards, episode lifecycle.
- `RL_EnemyController` (`Assets/Script/RL Scripts/Controller/RL_EnemyController.cs`) owns combat/HP/animation/death and calls back into the agent (`HandleDamage()`, `HandleEnemyDeath()`).
- `RL_TrainingManager` (`Assets/Script/RL Scripts/Training/RL_TrainingManager.cs`) orchestrates episode resets and counts active agents.
- `RL_TrainingEnemySpawner` and `RL_TrainingPlayerSpawner` manage multi-arena spawning; the player spawner configures curriculum bounds on spawned targets.
- `RL_CurriculumPlayerController` reads `Academy.Instance.EnvironmentParameters.GetWithDefault("player_difficulty", 0f)` and changes target behavior by stage.
- HCA adds a second sensor path: `ManagerObservationSensorComponent` -> `ManagerObservationSensor` (16 global features), while worker observations stay in `NormalEnemyAgent` (24 local features).
- `ManagerObservationSensorComponent` and `ManagerObservationSensor` (in `Assets/Script/RL Scripts/Normal Enemy/`) provide global state information for HCA critic networks.

### Training/inference workflow (project-specific)
- Environment/package baseline: Unity package `com.unity.ml-agents` is pinned in `Packages/manifest.json`; Python trainer entrypoint is `mlagents-learn` (also documented in `ml-agents/ml-agents/README.md`).
- Use the repo cheat sheet in `MLAgents COnda Script.txt` for canonical run patterns (`--resume`, `--force`, `--torch-device=cuda`).
- Typical runs:
  - PPO curriculum: `mlagents-learn config/ppo/NormalEnemyCC.yaml --run-id=PPO_Curriculum_v1`
  - HCA curriculum: `mlagents-learn config/hca/NormalEnemyHCA.yaml --run-id=HCA_Curriculum_v1`
  - Compare logs: `tensorboard --logdir=results`
- Build scenes include `Assets/Level/Scenes/Game Stage/Reinforcement Learning Stage.unity` (`ProjectSettings/EditorBuildSettings.asset`).

### Conventions that matter here
- `NormalEnemyAgent.TrainingActive` gates reset/death semantics across files; keep this contract intact when changing lifecycle logic (`NormalEnemyAgents.cs`, `RL_EnemyController.cs`, `RL_TrainingEnemySpawner.cs`).
- Agent reset path expects deactivated agents to be reactivated and `EndEpisode()`-driven; avoid destroy/deactivate behavior during training unless all reset callsites are updated.
- Reward tuning is centralized in `NormalEnemyRewards.cs`; use helper methods instead of scattering raw `AddReward()` calls.
- Spawner logic depends on patrol point ordering by name (`A->B->C->D`) and per-arena parent transforms; preserve these assumptions when editing spawn code.
- `RL_EnemyController` supports both `RL_PlayerController` and `PlayerController`; keep dual-path damage handling to avoid training/runtime regressions.
- HCA training uses `ManagerObservationSensorComponent` for 16 global features; when modifying observations, ensure both worker (24 local) and manager (16 global) sensor paths remain synchronized.
- Enemy stat observations (health/attack/speed) are normalized in `NormalEnemyAgent.CollectObservations()` to enable policy differentiation across enemy types.
- Performance-critical components cache references (`RL_TrainingPlayerSpawner`, `RL_TrainingManager`) in `Initialize()` to avoid `FindFirstObjectByType` calls every episode.
- Collision tracking uses layered counters (`obstacleCollisionCount`) with enter/stay/exit callbacks for accurate obstacle punishment.

### Integration points and external code boundaries
- HCA trainer implementation lives in vendored Python source: `ml-agents/ml-agents/mlagents/trainers/hca/{trainer.py,optimizer_torch.py}`.
- HCA config expects manager observation sensor index `manager_obs_index: -1`; if sensor order changes in prefabs, update YAML accordingly.
- Unity prefabs (`Assets/Level/RL Scenes/Training/Art/Models/RL Agents/*.prefab`) define action/observation sizes and decision cadence (`DecisionPeriod: 5`); keep code and prefab settings synchronized.

### Safe-edit checklist for agents
- Confirm behavior name, observation dimensions, and action spec stay consistent across C# + prefab + YAML.
- If changing episode/death flow, test both training mode (`TrainingActive=true`) and game mode (`TrainingActive=false`).
- If editing spawners, verify bounds/collision checks still produce non-zero spawns per arena.
- No dedicated Unity test suite was found under `Assets/**/*Test*.cs`; validate changes by focused play-mode training smoke runs.
