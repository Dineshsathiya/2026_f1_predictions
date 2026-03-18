:

🏎️ F1 Race Prediction Model (Python + FastF1)

A machine learning project that predicts the finishing order of a Formula 1 race using historical race data, driver performance metrics, and team performance.

This project uses FastF1, Python, and XGBoost to analyze previous race sessions and predict results for a target race (currently the 2026 Chinese Grand Prix).

📊 Features Used in the Model

The model evaluates multiple factors that influence race performance:

Driver Performance

Average lap time

Lap time consistency (standard deviation)

Lap time variance

Average stint pace

Total laps completed in previous races

Tire Behaviour

Tire degradation calculated from stint performance

Pace drop between early and late laps in a stint

Team / Constructor Form

Average team pace from previous races

Current constructor performance

Track Characteristics

Track downforce score

Sector pace averages

Qualifying Performance

Best qualifying time from Q1 / Q2 / Q3

🧠 Machine Learning Model

The prediction model uses:

XGBoost Regressor

The model learns from previous race pace and predicts race pace for the upcoming race.

Drivers are then ranked based on predicted race pace.

📦 Technologies Used

Python

FastF1

Pandas

NumPy

Scikit-Learn

XGBoost

Tkinter (for result visualization)

🖥️ Output

The script displays results in a popup window showing:

🏆 Predicted Podium

🥇 P1
🥈 P2
🥉 P3

📊 Top 10 Finishers
📋 Full Predicted Order
📂 Project Structure
F1-Race-Predictor/
│
├── china_gp_predictor.py
├── README.md
├── requirements.txt
└── f1_cache/
⚙️ Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/F1-Race-Predictor.git
cd F1-Race-Predictor

Install required packages:

pip install fastf1 pandas numpy scikit-learn xgboost
▶️ Running the Project

Run the script:

python china_gp_predictor.py

The program will:

Download race data using FastF1

Train the prediction model

Predict the race order

Display the results in a popup window

📈 Example Prediction Output
🏁 Predicted Chinese Grand Prix Order

1. VER | Max Verstappen | Red Bull
2. NOR | Lando Norris | McLaren
3. LEC | Charles Leclerc | Ferrari
...
⚠️ Notes

The model is a learning project, not a full race simulator.

Predictions do not account for:

Safety cars

DNFs

Weather changes

Strategy calls

Results depend on available race data in FastF1.

🚀 Future Improvements

Planned enhancements:

FP2 long-run pace analysis

Sprint race data integration

Pit stop performance

Weather prediction integration

Tire compound modeling

Track-specific performance modeling

Race strategy simulation

📚 Data Source

Race data is collected using the FastF1 API:

https://docs.fastf1.dev

🤝 Contributing

Contributions are welcome!

Feel free to:

open issues

submit pull requests

suggest new features

📄 License

MIT License
