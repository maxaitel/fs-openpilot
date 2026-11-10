# Execution setup record

The isolated same-image container stores this run in a 48 GiB RAM-backed filesystem. Original datasets, checkpoints and licensed asset sources remain read-only. Training source files were copied and checked against all 718 local snapshot entries. The exact source archive digest, calibration and protocol bind the actual training run.

Two initial calibration launches stopped with zero optimizer updates. The image lacked the optional Typer CLI dependency, so the launcher invokes the same lab_train.main parser directly. The first direct launch rejected a copied recovery-recipe path because gate receipts bind the original absolute path. The next launch used that original read-only path. Calibration then matched the unchanged model's prior plan and action metrics exactly. No thresholds or target values changed.

The first two camera pilot attempts stopped before driving. The scene exporter rejected a texture path resolving through the workspace asset symlink to the original external asset root. Their incomplete directories and logs are retained as a setup failure, not a driving DNF. The second attempt identified the panorama directory as another external symlink. The third attempt uses byte-identical local copies of all 28 material/person asset files (196,752,353 bytes) and 20 panorama files (41,557,386 bytes). The original assets and original symlink remain. The exporter guard is unchanged.

The exact setup records are in head_execution_setup.json, head_asset_copy.json and head_panorama_copy.json. These operational retries did not tune model hyperparameters, change the evaluation cases, or select a different checkpoint.
