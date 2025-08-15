# ⚡ BDSS Energy Optimizer

BDSS Energy Optimizer is an interactive **Business Decision Support System** built with **Streamlit** to plan cost-efficient daily energy usage under electricity price uncertainty.  
It uses **stochastic optimization** with scenario-based price data to generate an optimal base schedule and scenario-specific adjustments, considering operational constraints and risk preferences.

---

## 🚀 Features
- Upload **Excel files** containing multiple price scenarios with probabilities.
- **Advanced parameters**: smoothness penalty, ramp limits, minimum base coverage, price sensitivity, CVaR confidence, robustness weight, and real-time flexibility.
- **Preset strategies** for:
  - Cost Minimization
  - Smooth Operation
  - Risk Averse
  - Balanced
- **Interactive visualizations**:
  - Base vs adjusted schedules with price overlay
  - Heatmap of scenario adjustments
  - Scenario cost comparison charts
- **Risk metrics**: Value at Risk (VaR), Conditional Value at Risk (CVaR), Worst-case cost.
- **Detailed tables**:
  - Baseline vs Optimized cost and savings (Table 6.1)
  - Scenario probabilities, costs, and expected cost (Table 6.2)

---

## 📂 Project Structure
.
├── main.py # Streamlit app entry point
├── solving.py # Stochastic optimization model (Pyomo)
├── plotting.py # Plotly visualizations
├── requirements.txt # Python dependencies
└── sample_data.xlsx # Example price data file



---

---

## 🛠 Installation
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/bdss-energy-optimizer.git
cd bdss-energy-optimizer

Install dependencies
pip install -r requirements.txt
Run the app

streamlit run main.py
📊 Usage
Upload an Excel file with:

Prices sheet — hourly prices for each scenario

Probability sheet — probability of each scenario

Choose a preset or customize parameters.

Click Run Optimization (runs automatically on parameter change).

Explore results in:

Schedule plots

Scenario analysis

Cost breakdown tables

Risk analysis metrics

📦 Dependencies
Python 3.8+
Streamlit
Pandas
NumPy
Plotly
Pyomo
A solver (e.g., Gurobi)

📜 License
This project is licensed under the MIT License.

🙌 Acknowledgements
Developed as part of a Capstone Project on Decision Support Systems for Energy Cost Optimization.
