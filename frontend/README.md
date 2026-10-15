# CARFAC Explorer frontend

本機 React / TypeScript / Vite 研究介面。上傳 WAV/FLAC 或建立雙音刺激，比較正常與修改 OHC 參數的 CARFAC 模型；所有圖表使用 API 回傳的實際運算資料。

以下指令從 `carfac-ephys` repository 根目錄開始。先安裝後端與前端相依套件，並建置前端：

```sh
uv sync --extra explorer
cd frontend
npm ci
npm test
npm run build
cd ..
.venv/bin/python -m carfac_ephys.explorer_api --port 8768
```

開啟 `http://127.0.0.1:8768`。API 後端會提供已建置的 `frontend/dist/`，不需要另一個前端 server。

開發時可在第二個 terminal 使用 Vite（起始工作目錄仍為 repository 根目錄）：

```sh
cd frontend
npm run dev
```

Vite 在 `http://127.0.0.1:5173`，將 `/api` 代理至已啟動的 `http://127.0.0.1:8768`。

- 設定改動不會自動重新跑實驗；按 Run 才送出工作，既有結果會標示為過期。
- BM/source 熱圖各自使用跨條件共用的 80 dB 色階；source component 切換只在瀏覽器進行。
- 單 channel 波形為分析起點後的 20 ms 原始取樣，可點擊熱圖或選擇 CF。
- Reference 先呈現實測資料；benchmark 的兩條模型曲線共用正常模型 anchor；實測相對振幅減去自身第一點，實測與模型絕對單位分開。實測原始 unwrapped phase 與模型 cosine-reference wrapped phase 分開繪圖，不對稀疏模型點自動 unwrap。
- 完成的 Explore／Reference 結果可下載 JSON；下載連結始終指向目前顯示結果的 job ID，包含該次設定。
- OHC health 是模型參數，並非細胞存活比例。Source 與逆向輸出是未校準的模型量，不能直接當成物理 OHC 力或耳道聲壓。

「路徑驗證 Validation」頁面（`/?tab=validation`）比較 direct sum、delay only、filter only、full 四條路徑。每個刺激共用同一份 sources，固定 OHC health=1、無 reflection，四條路徑共用固定的逆向中耳 transfer；可選設定只控制共用前向中耳濾波；這個試驗不使用其他頁面的 OHC slider。四個等 log 頻帶交替 fit/test，每條路徑只在 fit 點估計一個全域 dB offset，再列出保留頻帶的 RMSE/MAE。完整 386 點觀測保持原樣，48 個模型刺激點直接挑選原始頻率，不插值觀測。

Validation 會讀取 API 保存的最近完成結果，重新整理仍可查看與下載；設定變動後仍顯示舊結果並標示需重跑。對齊後數值不是物理 dB SPL；phase reference 未對齊，不計算 phase error 或 latency，也不把振幅誤差排名當成生理 pass/fail。

Dependencies are pinned by `package-lock.json`; Plotly is loaded only when a real plot is rendered. No cloud service is used for audio or model computation.
