function showDialog(message) {
    const dialogOverlay = document.getElementById('dialogOverlay');
    const dialogMessage = document.getElementById('dialogMessage');

    dialogMessage.textContent = message;
    dialogOverlay.style.visibility = 'visible';
    dialogOverlay.classList.add('show');
}

document.getElementById("submitbtn").addEventListener("click", (event) => {
    event.preventDefault(); // Prevent default form submission behavior
    predictDiabetes();
    document.querySelector(".body").classList.add("hidden")
});

const predictDiabetes = async () => {
    // Define the input payload according to your specification
    const inputPayload = {
        a: Number(document.getElementById("question1").value),
        b: Number(document.getElementById("question2").value),
        c: Number(document.getElementById("question3").value),
        d: Number(document.getElementById("question4").value),
        e: Number(document.getElementById("question5").value),
        f: Number(document.getElementById("question6").value),
        g: Number(document.getElementById("question7").value),
        h: Number(document.getElementById("question8").value),
        i: Number(document.getElementById("question9").value),
        j: Number(document.getElementById("question10").value),
        k: Number(document.getElementById("question11").value),
        l: Number(document.getElementById("question12").value),
        m: Number(document.getElementById("question13").value),
        n: Number(document.getElementById("question14").value),
        o: Number(document.getElementById("question15").value),
        p: Number(document.getElementById("question16").value),
        q: Number(document.getElementById("question17").value),
        r: Number(document.getElementById("question18").value),
        s: Number(document.getElementById("question19").value),
        t: Number(document.getElementById("question20").value),
        u: Number(document.getElementById("question21").value),
        v: Number(document.getElementById("question22").value),
        w: Number(document.getElementById("question23").value),
        x: Number(document.getElementById("question24").value),
        y: Number(document.getElementById("question25").value),
        z: Number(document.getElementById("question26").value),
        a1: Number(document.getElementById("question27").value),
        b1: Number(document.getElementById("question28").value),
        c1: Number(document.getElementById("question29").value),
        d1: Number(document.getElementById("question30").value),
    };

    console.log(inputPayload);
    try {
        // Make the API call using fetch
        const response = await fetch(
            "https://ai-healthcare-webapp.onrender.com/predict/breastcancer",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(inputPayload),
            }
        );

        // Check if the response is ok
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        // Parse the JSON response
        const data = await response.json();
        console.log("Prediction Response:", data);

        // Display the message in the dialog box
        showDialog(data.prediction);  // Assuming the API response has a 'message' field

    } catch (error) {
        // Handle any errors that occur during the fetch
        console.error("Error making the API call:", error);
        showDialog("An error occurred while making the API call. Please try again."); // Show error message
    }
};
