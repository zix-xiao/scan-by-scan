import sys
import os

module_path = "/mnt/cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan"
if module_path not in sys.path:
    sys.path.append(module_path)
from prediction.RT import generate_reference
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
for_train_filepath_list = [
    "/mnt/cmnfs/proj/ORIGINS/data/brain/txt_ssDDA_LFQ_noMBR/evidence_fresh1.txt",
    "/mnt/cmnfs/proj/ORIGINS/data/brain/txt_ssDDA_LFQ_noMBR/evidence_fresh2.txt",
    "/mnt/cmnfs/proj/ORIGINS/data/brain/txt_ssDDA_LFQ_noMBR/evidence_fresh3.txt",
]
to_pred_filepath = "/mnt/cmnfs/proj/ORIGINS/data/brain/txt_3x13Brainregions_MBR_LFQ_iBAQ/evidence_freshfrozen_modseq_charge.txt"

for for_train_filepath in for_train_filepath_list:
    train_file_name = os.path.basename(for_train_filepath[:-4])
    exp_num = os.path.basename(for_train_filepath)[-5]
    print(for_train_filepath, exp_num)
    generate_reference(
        for_train_filepath=for_train_filepath,
        to_pred_filepath=to_pred_filepath,
        seed=42,
        pred_suffix=str(exp_num) + "_FilteredByClosestRT",
        filter_by_RT_diff="closest",
        save_model_name=train_file_name,
    )
