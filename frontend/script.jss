// ============================================================
// MEDVISION AI - FRONTEND JAVASCRIPT
// ============================================================

const fileInput = document.getElementById("image");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const loader = document.getElementById("loader");
const predictionBars = document.getElementById("predictionBars");
const report = document.getElementById("report");
const gradcamImage = document.getElementById("gradcamImage");
const downloadPDF = document.getElementById("downloadPDF");


// ============================================================
// IMAGE SELECTION
// ============================================================

fileInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        return;
    }

    // Check that selected file is an image
    if (!file.type.startsWith("image/")) {

        alert("Please select a valid X-ray image.");

        this.value = "";

        return;
    }

    // Create temporary browser URL
    const imageURL = URL.createObjectURL(file);

    // Display selected image
    preview.src = imageURL;

    // Make image visible
    preview.style.display = "block";

    // Change upload button text
    const uploadBox = document.querySelector(".upload-box");

    if (uploadBox) {

        uploadBox.classList.add("selected");

        uploadBox.querySelector("h3").textContent =
            "✓ X-Ray Selected";

        uploadBox.querySelector("p").textContent =
            file.name;
    }

    // Enable prediction button
    predictBtn.disabled = false;

});


// ============================================================
// DRAG & DROP
// ============================================================

const uploadBox = document.querySelector(".upload-box");

if (uploadBox) {

    uploadBox.addEventListener("dragover", function (event) {

        event.preventDefault();

        uploadBox.classList.add("dragover");

    });


    uploadBox.addEventListener("dragleave", function () {

        uploadBox.classList.remove("dragover");

    });


    uploadBox.addEventListener("drop", function (event) {

        event.preventDefault();

        uploadBox.classList.remove("dragover");

        const files = event.dataTransfer.files;

        if (!files || files.length === 0) {
            return;
        }

        const file = files[0];

        if (!file.type.startsWith("image/")) {

            alert("Please drop a valid image file.");

            return;
        }

        // Put dropped file into file input
        const dataTransfer = new DataTransfer();

        dataTransfer.items.add(file);

        fileInput.files = dataTransfer.files;

        // Trigger normal image selection
        fileInput.dispatchEvent(new Event("change"));

    });

}


// ============================================================
// PREDICT
// ============================================================

async function predict() {

    const file = fileInput.files[0];

    if (!file) {

        alert("Please select an X-ray image first.");

        return;
    }


    // Show loader
    loader.classList.remove("hidden");


    // Clear previous results
    predictionBars.innerHTML = "";
    report.innerHTML = "";

    // Hide previous Grad-CAM
    gradcamImage.style.display = "none";


    // Create FormData
    const formData = new FormData();

    formData.append("image", file);


    try {

        console.log("Sending image to Flask backend...");
        console.log("File:", file.name);
        console.log("Size:", file.size);
        console.log("Type:", file.type);


        // Send image to Flask
        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",
                body: formData
            }
        );


        console.log("Server response:", response.status);


        // Check server response
        if (!response.ok) {

            const errorText = await response.text();

            throw new Error(
                "Server returned " +
                response.status +
                ": " +
                errorText
            );
        }


        // Convert response to JSON
        const data = await response.json();


        console.log("Prediction received:", data);


        // Display prediction bars
        displayPredictions(data);


        // Show Grad-CAM
        showGradCAM();


        // Hide loader
        loader.classList.add("hidden");


    } catch (error) {

        console.error("Prediction error:", error);

        loader.classList.add("hidden");

        alert(
            "Prediction failed.\n\n" +
            error.message +
            "\n\nMake sure Flask is running at:\n" +
            "http://127.0.0.1:5000"
        );

    }

}


// ============================================================
// DISPLAY PREDICTIONS
// ============================================================

function displayPredictions(data) {

    predictionBars.innerHTML = "";

    const entries = Object.entries(data)
        .filter(function ([disease, score]) {

            return typeof score === "number";

        })
        .sort(function (a, b) {

            return b[1] - a[1];

        });


    if (entries.length === 0) {

        predictionBars.innerHTML =
            "<p>No prediction results received.</p>";

        return;
    }


    const heading = document.createElement("h2");

    heading.textContent = "Disease Confidence";

    predictionBars.appendChild(heading);


    entries.forEach(function ([disease, score], index) {

        const percentage = Math.max(
            0,
            Math.min(100, score * 100)
        );


        const card = document.createElement("div");

        card.className = "disease-card";


        card.innerHTML = `

            <div class="disease-name">

                <span>${formatDiseaseName(disease)}</span>

                <strong>
                    ${percentage.toFixed(1)}%
                </strong>

            </div>


            <div class="progress">

                <div
                    class="progress-fill"
                    style="width: 0%"
                ></div>

            </div>

        `;


        predictionBars.appendChild(card);


        // Animate bar after it enters DOM
        setTimeout(function () {

            const bar =
                card.querySelector(".progress-fill");

            bar.style.width =
                percentage + "%";

        }, 100 + index * 80);

    });


    // Create top finding
    const topDisease = entries[0][0];

    const topScore = entries[0][1] * 100;


    report.innerHTML = `

        <div class="top-finding">

            <div class="finding-label">
                PRIMARY AI FINDING
            </div>

            <div class="finding-disease">
                ${formatDiseaseName(topDisease)}
            </div>

            <div class="finding-confidence">
                ${topScore.toFixed(1)}% confidence
            </div>

        </div>

    `;

}


// ============================================================
// FORMAT DISEASE NAME
// ============================================================

function formatDiseaseName(name) {

    return name.replace(/_/g, " ");

}


// ============================================================
// GRAD-CAM
// ============================================================

function showGradCAM() {

    // Cache-busting timestamp
    const timestamp = new Date().getTime();

    gradcamImage.src =
        "gradcam.png?" + timestamp;

    gradcamImage.onload = function () {

        gradcamImage.style.display = "block";

    };


    gradcamImage.onerror = function () {

        console.log(
            "Grad-CAM image not available yet."
        );

    };

}


// ============================================================
// PDF DOWNLOAD
// ============================================================

downloadPDF.addEventListener("click", async function () {

    try {

        const response = await fetch(
            "http://127.0.0.1:5000/download-report"
        );


        if (!response.ok) {

            throw new Error(
                "PDF download endpoint is not available."
            );

        }


        const blob = await response.blob();


        const url =
            window.URL.createObjectURL(blob);


        const link =
            document.createElement("a");


        link.href = url;

        link.download =
            "MEDVISION_AI_Report.pdf";


        document.body.appendChild(link);

        link.click();

        link.remove();


        window.URL.revokeObjectURL(url);


    } catch (error) {

        console.error(
            "PDF download error:",
            error
        );


        alert(
            "PDF download is not connected yet."
        );

    }

});


// ============================================================
// INITIAL STATE
// ============================================================

predictBtn.disabled = true;

preview.style.display = "none";

gradcamImage.style.display = "none";


// ============================================================
// CONSOLE MESSAGE
// ============================================================

console.log(
    "%cMEDVISION AI",
    "font-size:24px;font-weight:bold;color:#00c6ff;"
);

console.log(
    "Frontend initialized successfully."
);