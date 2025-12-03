async function sendImage() {
    const fileInput = document.getElementById("imageUpload");

    // If no image is selected, hide results and stop
    if (fileInput.files.length === 0) {
        alert("Please upload an image first.");
        document.getElementById("results").style.display = "none";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // Send image to backend
    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    // Fill YOLO results
    document.getElementById("yolo-class").textContent = data.model1_class;
    document.getElementById("yolo-health").textContent = data.model1_health;

    // Fill MobileNet results
    document.getElementById("mobile-class").textContent = data.model2_class;
    document.getElementById("mobile-health").textContent = data.model2_health;

    // Show results only after receiving predictions
    document.getElementById("results").style.display = "flex";
}
