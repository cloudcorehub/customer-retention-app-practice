const form = document.getElementById("prediction-form");
const emptyState = document.getElementById("empty-state");
const result = document.getElementById("result");
const errorMessage = document.getElementById("error-message");
const modelStatus = document.getElementById("model-status");

async function checkModelStatus() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error();
    modelStatus.textContent = "● model ready";
    modelStatus.className = "status-dot";
  } catch (_error) {
    modelStatus.textContent = "● model unavailable";
    modelStatus.className = "status-dot unavailable";
  }
}

checkModelStatus();

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    tenure_months: Number(document.getElementById("tenure_months").value),
    monthly_charges: Number(document.getElementById("monthly_charges").value),
    contract: document.getElementById("contract").value,
    internet_service: document.getElementById("internet_service").value,
    tech_support: document.getElementById("tech_support").value,
    payment_method: document.getElementById("payment_method").value,
  };

  const button = form.querySelector("button");
  button.disabled = true;
  button.textContent = "Running prediction…";
  errorMessage.classList.add("hidden");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) throw new Error("Prediction request failed");
    const data = await response.json();

    emptyState.classList.add("hidden");
    result.classList.remove("hidden");

    const percent = Math.round(data.churn_probability * 100);
    document.getElementById("probability").textContent = `${percent}%`;
    document.getElementById("meter-fill").style.width = `${percent}%`;
    document.getElementById("model-type").textContent = data.model_type;
    document.getElementById("model-version").textContent = data.model_version;
    document.getElementById("app-version").textContent = data.app_version;

    const badge = document.getElementById("risk-badge");
    badge.textContent = `${data.risk_level} risk`;
    badge.className = `risk-badge ${data.risk_level.toLowerCase()}`;
  } catch (error) {
    errorMessage.textContent = `${error.message}. Check that the API is running, then try again.`;
    errorMessage.classList.remove("hidden");
  } finally {
    button.disabled = false;
    button.textContent = "Predict churn risk";
  }
});
