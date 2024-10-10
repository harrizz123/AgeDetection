import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.highgui.HighGui;

public class AgeDetection {

    // Pre-defined age groups based on the model's training
    private static final String[] AGE_GROUPS = new String[]{
            "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"
    };

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the image
        String imagePath = "K:\\Training\\AgeDetection\\src\\main\\resources\\image.jpg";
        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            System.out.println("Could not open or find the image!");
            return;
        }

        // Step 1: Load Haar Cascade for face detection
        CascadeClassifier faceDetector = new CascadeClassifier("K:\\Training\\AgeDetection\\haarcascade_frontalface_default.xml");
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(image, faceDetections);

        // Step 2: Load pre-trained age detection model
        String modelProto = "K:\\Training\\AgeDetection\\src\\main\\resources\\models\\age_deploy.prototxt"; // Path to prototxt file
        String modelWeights = "K:\\Training\\AgeDetection\\src\\main\\resources\\models\\age_net.caffemodel"; // Path to caffemodel file
        Net ageNet = Dnn.readNetFromCaffe(modelProto, modelWeights);

        // Step 3: Loop through each detected face
        for (Rect rect : faceDetections.toArray()) {
            // Draw a rectangle around the face
            Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));

            // Extract the face ROI and preprocess it for the age model
            Mat faceROI = new Mat(image, rect);
            Mat blob = Dnn.blobFromImage(faceROI, 1.0, new Size(227, 227), new Scalar(104, 117, 123), false, false);

            // Step 4: Predict the age
            ageNet.setInput(blob);
            Mat agePredictions = ageNet.forward();
            Core.MinMaxLocResult result = Core.minMaxLoc(agePredictions);
            int ageIndex = (int) result.maxLoc.x;
            String predictedAge = AGE_GROUPS[ageIndex];

            // Step 5: Add the predicted age to the image
            Imgproc.putText(image, predictedAge, new Point(rect.x, rect.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(255, 0, 0), 2);
        }

        // Step 6: Display the final image with detected faces and predicted ages
        HighGui.imshow("Age Detection", image);
        HighGui.waitKey(0);
    }
}
