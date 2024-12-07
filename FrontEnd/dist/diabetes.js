function showDialog(message) {
    const dialogOverlay = document.getElementById('dialogOverlay');
    const dialogMessage = document.getElementById('dialogMessage');

    dialogMessage.textContent = message;
    dialogOverlay.style.visibility = 'visible';
    dialogOverlay.classList.add('show');
}

// Variables to hold form pages and buttons
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
        predictDiabetes();
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

// Function to update the progress bar based on current page
function updateProgressBar() {
    const progress = ((currentPage + 1) / pages.length) * 100;
    progressBar.style.width = progress + "%";
}

// Function to toggle the visibility and text of navigation buttons
function toggleButtons() {
    prevBtn.disabled = currentPage === 0; // Disable Previous button on first page
    nextBtn.textContent = currentPage === pages.length - 1 ? "Submit" : "Next"; // Change Next button to Submit on last page
}

// Function to collect data and submit the form
const predictDiabetes = async () => {
    // Define the input payload according to your specification
    const inputPayload = {
        Pregnancies: document.getElementById("input1").value,
        Glucose: document.getElementById("input2").value,
        BloodPressure: document.getElementById("input3").value,
        SkinThickness: document.getElementById("input4").value,
        Insulin: document.getElementById("input5").value,
        BMI: document.getElementById("input6").value,
        DiabetesPedigreeFunction: document.getElementById("input7").value,
        Age: document.getElementById("input8").value,
    };

    console.log(inputPayload);

    try {
        // Make the API call using fetch
        const response = await fetch("https://ai-healthcare-webapp.onrender.com/predict/diabetes", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(inputPayload),
        });

        // Check if the response is ok
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        // Parse the JSON response
        const data = await response.json();
        console.log("Prediction Response:", data);

        // Show the dialog with the API response message
        // Assuming the API response has a 'message' field
        showDialog(data.prediction);
    } catch (error) {
        // Handle any errors that occur during the fetch
        console.error("Error making the API call:", error);
        showDialog("An error occurred while making the prediction. Please try again.");
    }
};
