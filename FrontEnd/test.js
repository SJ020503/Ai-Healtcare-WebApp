// diabetes_predict.js
const predictDiabetes = async () => {
  // Define the input payload according to your specification
  const dat = {
    a: 14.2,
    b: 20.6,
    c: 92.7,
    d: 620.0,
    e: 0.096,
    f: 0.114,
    g: 0.085,
    h: 0.055,
    i: 0.185,
    j: 0.06,
    k: 0.07,
    l: 1.2,
    m: 2.3,
    n: 8.1,
    o: 75.0,
    p: 0.007,
    q: 0.032,
    r: 0.029,
    s: 0.016,
    t: 0.019,
    u: 0.003,
    v: 24.3,
    w: 34.5,
    x: 155.0,
    y: 1300.0,
    z: 0.162,
    a1: 0.256,
    b1: 0.29,
    c1: 0.12,
    d1: 0.265,
  };

  console.log(dat);
  try {
    // Make the API call using fetch
    const response = await fetch(
      "https://ai-healthcare-webapp.onrender.com/predict/breastcancer",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(dat),
      }
    );

    // Check if the response is ok
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    // Parse the JSON response
    const data = await response.json();
    console.log("Prediction Response:", data);
  } catch (error) {
    // Handle any errors that occur during the fetch
    console.error("Error making the API call:", error);
  }
};

// Call the function to make the API request
predictDiabetes();
