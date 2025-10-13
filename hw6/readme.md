Role:
You are an expert machine learning developer building a fully browser-based, GitHub Pages-deployable stock prediction demo. Everything must run client-side with TensorFlow.js. The solution must load market data from a given CSV file (like "sp500_top10_xcorr_recent3y.csv"), train a multi-output GRU model, and provide clear, interactive visualization of binary classification results for multiple stocks.

Task:
- Read the user's local CSV file (containing Date, Symbol, Open, Close for 10 S&P 500 stocks, daily).
- Prepare a time series dataset so that for each sample, the input is a 12-day sequence of features for all 10 stocks, and the output is a 3-day-ahead binary up/down classification (Close price rising vs falling, per stock).
- Design the model using GRU layers in TensorFlow.js, with output shape (10 stocks × 3 days = 30 binaries).
- Train this model entirely in-browser.
- After prediction, compute per-stock accuracy, rank the 10 stocks by accuracy, and visualize results with sorted accuracy bar charts and per-stock prediction timelines (correct/wrong).
- All code must be organized into these three JS modules: data-loader.js, gru.js (model definition/training), app.js (UI, visualizations).

Instruction:
- Index.html (not included here) should have UI to upload the CSV, launch training, show progress, view accuracy.
- data-loader.js: Parse local CSV (via file input). Pivot to align dates and symbols, select "Open" and "Close" as features, normalize (MinMax per stock), and prepare sliding-window samples:
  - Input: For each date D, provide last 12 days' [Open, Close] for all 10 symbols (shape: [samples, 12, 20]).
  - Output: For each sample, compute “target binary label” for each stock for days D+1, D+2, D+3:
    - Label = 1 if Close(t+offset) > Close(D) else 0, for offset=1,2,3, per stock.
  - Split samples chronologically into train/test. Export tensors: X_train, y_train, X_test, y_test, and mapping of stock symbols.
- gru.js: Build and compile multi-output GRU model for binary classification:
  - Input: shape = (12, 20)
  - Stacked GRU (or bidirectional) layers, then Dense 30 (sigmoid) for 10 stocks × 3 steps.
  - Loss: binaryCrossentropy; metrics: binaryAccuracy (also compute accuracy by stock after inference).
  - Provide fit, predict, and evaluation utilities. Allow for saving and reloading weights if desired.
- app.js: UI control and visualization:
  - Tie UI to data, model, and training flow.
  - On evaluation, compute accuracy for each stock across all test samples (averaged over 3 output days).
  - Sort stocks by accuracy, render a horizontal bar chart of accuracies (best to worst).
  - For each stock, plot a colored timeline of prediction results (green/red for correct/wrong, timeline is day axis).
  - Display confusion matrix for each stock (optional).
- All JS files must use tf.js from CDN and ES6 classes/modules; all dependencies must be client-side.
- Code must handle memory disposal, edge/corner cases, and robust error handling for file loading and shape mismatches.
- Use clear English comments.
- Designed for direct deployment on GitHub Pages (no server or Python backend).

Format:
- Output three code blocks labeled exactly as: data-loader.js, gru.js, app.js along with index.html.
- No explanations, only code inside the code blocks.
