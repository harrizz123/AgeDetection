import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

public class AgeDetectionLive {

    // Pre-defined age groups based on the model's training
    private static final String[] AGE_GROUPS = new String[]{
            "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"
    };

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Step 1: Initialize the video capture (default camera index is 0)
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("Error: Could not open the camera!");
            return;
        }

        // Step 2: Load Haar Cascade for face detection
        CascadeClassifier faceDetector = new CascadeClassifier("K:\\Training\\AgeDetection\\haarcascade_frontalface_default.xml");

        // Step 3: Load pre-trained age detection model
        String modelProto = "K:\\Training\\AgeDetection\\src\\main\\resources\\models\\age_deploy.prototxt"; // Path to prototxt file
        String modelWeights = "K:\\Training\\AgeDetection\\src\\main\\resources\\models\\age_net.caffemodel"; // Path to caffemodel file
        Net ageNet = Dnn.readNetFromCaffe(modelProto, modelWeights);

        // Step 4: Continuously capture video frames
        Mat frame = new Mat();
        while (camera.read(frame)) {
            // Convert the frame to grayscale
            Mat grayFrame = new Mat();
            Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

            // Detect faces in the frame
            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(grayFrame, faceDetections);

            // Process each detected face
            for (Rect rect : faceDetections.toArray()) {
                // Draw a rectangle around the face
                Imgproc.rectangle(frame, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));

                // Extract the face ROI and preprocess it for the age model
                Mat faceROI = new Mat(frame, rect);
                Mat blob = Dnn.blobFromImage(faceROI, 1.0, new Size(227, 227), new Scalar(104, 117, 123), false, false);

                // Predict the age
                ageNet.setInput(blob);
                Mat agePredictions = ageNet.forward();
                Core.MinMaxLocResult result = Core.minMaxLoc(agePredictions);
                int ageIndex = (int) result.maxLoc.x;
                String predictedAge = AGE_GROUPS[ageIndex];

                // Add the predicted age to the frame
                Imgproc.putText(frame, predictedAge, new Point(rect.x, rect.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(255, 0, 0), 2);
            }

            // Display the frame
            HighGui.imshow("Age Detection - Live Camera", frame);

            // Break the loop if 'q' is pressed
            if (HighGui.waitKey(30) == 'q') {
                break;
            }
        }

        // Release the camera and close the window
        camera.release();
        HighGui.destroyAllWindows();
    }
}
