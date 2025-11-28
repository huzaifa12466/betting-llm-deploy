async function askQuestion() {
    const question = document.getElementById("question").value;
    const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question })
    });
    const data = await response.json();
    document.getElementById("answer").innerText = data.answer;
}
