# LangChainPlusLlama2_test

這是一個在自己電腦上，搭建Llama2讓其將PDF檔當作知識庫的測試

這只是一個簡單測試的專案

## 安裝以及注意事項

建議創建虛擬環境

```
python -m venv .\venv
```

接著安裝需要的儲存庫

```
pip install -r requirements.txt
```

如果在安裝 `llama-cpp-python` 時遇到錯誤，代表電腦未安裝 C++ 的編譯器，建議可透過 Visual Studio 安裝。

若依舊無法成功編譯，或是編譯出來後，無法調用 GPU，可以嘗試以下指令，安裝別人編譯好的檔案 (請自行修改中間cuda toolkit的版本):
```
python -m pip install llama-cpp-python --prefer-binary --no-cache-dir --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117 --force-reinstall
```

## 設定

再 `.env` 檔案中，我們可以簡單的設定一下要使用的模型，以及 gguf 模型載入 GPU 的層數。

目前可設定東西如下:
```shell
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE="cuda:0"
LLM_MODEL_PATH="./model/chinese-alpaca-2-7b.Q4_K_M.gguf"

N_GPU_LAYERS=15
```
