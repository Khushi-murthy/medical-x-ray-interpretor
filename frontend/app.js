// ============================================
// AI Medical X-Ray Interpreter
// Frontend JavaScript
// Part 1
// ============================================

// Image Input
const imageInput = document.getElementById("image");

// Preview Image
const preview = document.getElementById("preview");

// Result Div
const resultDiv = document.getElementById("result");

// Report Div
const reportDiv = document.getElementById("report");

// Loading
const loading = document.getElementById("loading");

// Drop Area
const dropArea = document.getElementById("dropArea");

// Chart
let predictionChart = null;


// ============================================
// Image Preview
// ============================================

imageInput.addEventListener("change", function(){

    const file = this.files[0];

    if(!file) return;

    preview.src = URL.createObjectURL(file);

    preview.style.display = "block";

});


// ============================================
// Drag & Drop
// ============================================

dropArea.addEventListener("dragover",(e)=>{

    e.preventDefault();

    dropArea.classList.add("dragover");

});

dropArea.addEventListener("dragleave",()=>{

    dropArea.classList.remove("dragover");

});

dropArea.addEventListener("drop",(e)=>{

    e.preventDefault();

    dropArea.classList.remove("dragover");

    imageInput.files = e.dataTransfer.files;

    const file = e.dataTransfer.files[0];

    preview.src = URL.createObjectURL(file);

    preview.style.display = "block";

});


// ============================================
// Predict
// ============================================

async function predict(){

    const file = imageInput.files[0];

    if(!file){

        alert("Please upload an X-Ray.");

        return;

    }

    loading.style.display = "block";

    resultDiv.innerHTML = "";

    reportDiv.innerHTML = "";

    const formData = new FormData();

    formData.append("image",file);

    const response = await fetch(

        "http://127.0.0.1:5000/predict",

        {

            method:"POST",

            body:formData

        }

    );

    const data = await response.json();
    document.getElementById("gradcamImage").src =
"gradcam.png?" + new Date().getTime();

document.getElementById("gradcamImage").style.display =
"block";

    loading.style.display = "none";

    displayPredictions(data.predictions);

    displayReport(data.report);

}

// ============================================
// Display Prediction Cards
// ============================================

function displayPredictions(predictions){

    const sorted = Object.entries(predictions)
        .sort((a,b)=>b[1]-a[1]);

    const topDisease = sorted[0][0];
    const topScore = sorted[0][1];

    let html = "";

    // ===========================
    // Top Card
    // ===========================

    html += `
    <div class="top-card fade-in">

        <h2>
            <i class="fa-solid fa-stethoscope"></i>
            ${topDisease}
        </h2>

        <p>

            Highest Confidence :
            <b>${(topScore*100).toFixed(2)}%</b>

        </p>

    </div>
    `;

    // ===========================
    // Prediction Grid
    // ===========================

    html += `<div class="prediction-grid">`;

    sorted.forEach(item=>{

        const disease = item[0];

        const score = item[1];

        let badge = "badge-low";

        let level = "Low";

        if(score>=0.60){

            badge="badge-high";

            level="High";

        }

        else if(score>=0.30){

            badge="badge-medium";

            level="Medium";

        }

        html += `

        <div class="prediction-card hover-lift">

            <div class="disease-title">

                ${disease}

            </div>

            <div class="percent">

                ${(score*100).toFixed(2)}%

            </div>

            <div class="progress">

                <div
                    class="progress-bar"
                    style="width:${score*100}%"
                ></div>

            </div>

            <div class="confidence">

                <span>

                    Confidence

                </span>

                <span class="${badge}">

                    ${level}

                </span>

            </div>

        </div>

        `;

    });

    html += "</div>";

    resultDiv.innerHTML = html;

    drawChart(sorted);

}

// ============================================
// AI Report
// ============================================

function displayReport(report){

    reportDiv.innerHTML="";

    reportDiv.classList.add("report-card");

    let index=0;

    const speed=15;

    function type(){

        if(index<report.length){

            reportDiv.innerHTML += report.charAt(index);

            index++;

            setTimeout(type,speed);

        }

    }

    type();

}

// ============================================
// Chart
// ============================================

function drawChart(data){

    const labels=[];

    const values=[];

    data.forEach(item=>{

        labels.push(item[0]);

        values.push((item[1]*100).toFixed(2));

    });

    const ctx=document.getElementById("chart");

    if(predictionChart){

        predictionChart.destroy();

    }

    predictionChart=new Chart(ctx,{

        type:"bar",

        data:{

            labels:labels,

            datasets:[{

                label:"Disease Probability (%)",

                data:values,

                borderWidth:2,

                borderRadius:12,

                backgroundColor:[
                    "#2F80ED",
                    "#56CCF2",
                    "#27AE60",
                    "#EB5757",
                    "#9B51E0",
                    "#F2994A",
                    "#219653",
                    "#6FCF97",
                    "#BB6BD9",
                    "#2D9CDB",
                    "#F2C94C",
                    "#8E44AD",
                    "#16A085",
                    "#2980B9"
                ]

            }]

        },

        options:{

            responsive:true,

            plugins:{

                legend:{

                    labels:{

                        color:"white"

                    }

                }

            },

            scales:{

                x:{

                    ticks:{

                        color:"white"

                    }

                },

                y:{

                    beginAtZero:true,

                    max:100,

                    ticks:{

                        color:"white"

                    }

                }

            }

        }

    });

}

// ============================================
// Toast Notification
// ============================================

function showToast(message){

    let toast = document.querySelector(".toast-card");

    if(!toast){

        toast = document.createElement("div");

        toast.className = "toast-card";

        document.body.appendChild(toast);

    }

    toast.innerHTML = message;

    toast.style.display = "block";

    setTimeout(()=>{

        toast.style.display="none";

    },3000);

}


// ============================================
// Download Report as PDF
// ============================================

function downloadReport(){

    window.open(

        "http://127.0.0.1:5000/download-report",

        "_blank"

    );

}


// ============================================
// Theme Toggle
// ============================================

const themeButton = document.querySelector(".theme-btn");

themeButton.addEventListener("click",()=>{

    document.body.classList.toggle("dark");

    const icon = themeButton.querySelector("i");

    if(document.body.classList.contains("dark")){

        icon.className="fa-solid fa-sun";

        showToast("Dark Mode Enabled");

    }

    else{

        icon.className="fa-solid fa-moon";

        showToast("Light Mode Enabled");

    }

});


// ============================================
// Image Popup
// ============================================

preview.addEventListener("click",()=>{

    if(preview.src=="") return;

    const popup=document.createElement("div");

    popup.style.position="fixed";

    popup.style.left="0";

    popup.style.top="0";

    popup.style.width="100%";

    popup.style.height="100%";

    popup.style.background="rgba(0,0,0,.8)";

    popup.style.display="flex";

    popup.style.justifyContent="center";

    popup.style.alignItems="center";

    popup.style.zIndex="99999";

    popup.innerHTML=`

        <img
        src="${preview.src}"
        style="
        max-width:80%;
        max-height:85%;
        border-radius:20px;
        box-shadow:0 30px 50px rgba(0,0,0,.5);
        ">

    `;

    popup.onclick=()=>popup.remove();

    document.body.appendChild(popup);

});


// ============================================
// Reset Prediction
// ============================================

function resetPrediction(){

    imageInput.value="";

    preview.src="";

    preview.style.display="none";

    resultDiv.innerHTML="";

    reportDiv.innerHTML="Waiting for analysis...";

    if(predictionChart){

        predictionChart.destroy();

    }

    showToast("Ready for another X-Ray");

}


// ============================================
// Keyboard Shortcut
// ============================================

document.addEventListener("keydown",(e)=>{

    if(e.key==="Escape"){

        resetPrediction();

    }

});


// ============================================
// Better Error Handling
// ============================================

window.addEventListener("unhandledrejection",(event)=>{

    loading.style.display="none";

    showToast("Something went wrong.");

});


// ============================================
// Loading Messages
// ============================================

const loadingMessages=[

"Initializing DenseNet121...",

"Reading Chest X-Ray...",

"Extracting Medical Features...",

"Running Disease Classification...",

"Generating AI Report...",

"Preparing Confidence Scores..."

];

let loadingInterval=null;

function startLoadingText(){

    const loadingText=document.querySelector("#loading p");

    let i=0;

    loadingInterval=setInterval(()=>{

        loadingText.innerHTML=loadingMessages[i];

        i++;

        if(i>=loadingMessages.length){

            i=0;

        }

    },1200);

}

function stopLoadingText(){

    clearInterval(loadingInterval);

}


// ============================================
// Update Predict Function
// ============================================

const originalPredict = predict;

predict = async function(){

    startLoadingText();

    try{

        await originalPredict();

        showToast("Analysis Completed Successfully");

    }

    catch(error){

        console.error(error);

        showToast("Prediction Failed");

    }

    stopLoadingText();

};


// ============================================
// Auto Fade Sections
// ============================================

window.addEventListener("load",()=>{

    document.querySelectorAll(".glass-card").forEach(card=>{

        card.classList.add("fade-in");

    });

});


// ============================================
// AI Chat Placeholder
// ============================================

function askAI(){

    const input=document.querySelector(".chat-input input");

    if(!input) return;

    if(input.value.trim()=="") return;

    showToast("AI Chat will be connected in the next step.");

    input.value="";

}


// ============================================
// Download Button
// ============================================

document.addEventListener("DOMContentLoaded",()=>{

    const report=document.getElementById("report");

    const btn=document.createElement("button");

    btn.innerHTML='<i class="fa-solid fa-file-pdf"></i> Download Report';

    btn.className="download-btn mt-4";

    btn.onclick=downloadReport;

    report.parentElement.appendChild(btn);

});


// ============================================
// End
// ============================================

console.log("AI Medical Frontend Loaded Successfully");