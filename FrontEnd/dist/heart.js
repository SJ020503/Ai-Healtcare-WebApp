function showDialog(message) {
    const dialogOverlay = document.getElementById('dialogOverlay');
    const dialogMessage = document.getElementById('dialogMessage');

    dialogMessage.textContent = message;
    dialogOverlay.style.visibility = 'visible';
    dialogOverlay.classList.add('show');
}

const pages = document.querySelectorAll(".form-page");
const nextBtn = document.getElementById("next-btn");
const prevBtn = document.getElementById("prev-btn");
const progressBar = document.getElementById("progress-bar");
let currentPage = 0;

// Event listener for Next button
nextBtn.addEventListener("click", () => {
    if (currentPage < pages.length - 1) {
        pages[currentPage].classList.add("hidden");
        currentPage++;
        pages[currentPage].classList.remove("hidden");
        updateProgressBar();
    } else {
        // When the last page is reached, submit the form
        predictHeartDisease();
    }
    toggleButtons();
});

// Event listener for Previous button
prevBtn.addEventListener("click", () => {
    if (currentPage > 0) {
        pages[currentPage].classList.add("hidden");
        currentPage--;
        pages[currentPage].classList.remove("hidden");
        updateProgressBar();
    }
    toggleButtons();
});

// Function to update the progress bar based on the current page
function updateProgressBar() {
    const progress = ((currentPage + 1) / pages.length) * 100;
    progressBar.style.width = progress + "%";
}

// Function to toggle the visibility and text of navigation buttons
function toggleButtons() {
    prevBtn.style.display = currentPage === 0 ? "none" : "block"; // Hide Previous button on the first page
    nextBtn.textContent = currentPage === pages.length - 1 ? "Submit" : "Next"; // Change Next button to Submit on the last page
}

// Function to collect data and submit the form
const predictHeartDisease = async () => {
    // Define the input payload according to your specification
    const inputPayload = {
        age: Number(document.getElementById("input1").value),
        sex: Number(document.querySelector('input[name="input2"]:checked')?.value),
        cp: Number(document.querySelector('input[name="input3"]:checked')?.value),
        trestbps: Number(document.getElementById("input4").value),
        chol: Number(document.getElementById("input5").value),
        fbs: Number(document.querySelector('input[name="input6"]:checked')?.value),
        restecg: Number(document.querySelector('input[name="input7"]:checked')?.value),
        thalach: Number(document.getElementById("input8").value),
        exang: Number(document.querySelector('input[name="input9"]:checked')?.value),
        oldpeak: Number(document.getElementById("input10").value),
        slope: Number(document.querySelector('input[name="input11"]:checked')?.value),
        ca: Number(document.getElementById("input12").value),
        thal: Number(document.querySelector('input[name="input13"]:checked')?.value),
    };

    console.log(inputPayload); // Check collected data
    try {
        // Make the API call using fetch
        const response = await fetch(
            "https://ai-healthcare-webapp.onrender.com/predict/heartdisease",
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

        // Show the dialog with the API response message
        showDialog(data.prediction); // Update this line to reflect how the API response is structured
    } catch (error) {
        // Handle any errors that occur during the fetch
        console.error("Error making the API call:", error);
        showDialog("An error occurred while making the prediction. Please try again.");
    }
};
