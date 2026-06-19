from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="/content/drive/My Drive/tourism_prediction/hosting",     # the local folder containing your files
    repo_id="vishaldixit75/tourismData",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
