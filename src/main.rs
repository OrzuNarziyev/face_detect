use opencv::{
    core::{Size, Rect2f, Scalar},
    highgui, imgcodecs, imgproc,
    objdetect::{FaceDetectorYN, FaceRecognizerSF, FaceRecognizerSF_DisType},
    prelude::*,
    videoio,
    Result,
};

fn main() -> Result<()> {
    // Load ONNX model files
    let detector_model = "models/face_detection_yunet_2021dec.onnx";
    let recognizer_model = "models/face_recognition_sface_2021dec.onnx";

    // Load known face image and extract feature
    let known_image = imgcodecs::imread("known_face.jpg", imgcodecs::IMREAD_COLOR)?;
    if known_image.empty() {
        panic!("❌ known_face.jpg not found!");
    }

    let mut detector = FaceDetectorYN::create(
        detector_model,
        "",
        Size::new(320, 320),
        0.9,
        0.3,
        5000,
        0,
        0,
    )?;
    detector.set_input_size(known_image.size()?)?;

    let mut recognizer = FaceRecognizerSF::create(recognizer_model, "", 0, 0)?;

    let mut known_faces = Mat::default();
    detector.detect(&known_image, &mut known_faces)?;

    if known_faces.rows() < 1 {
        panic!("❌ No face found in known_face.jpg");
    }

    let mut known_aligned = Mat::default();
    recognizer.align_crop(&known_image, &known_faces.row(0)?, &mut known_aligned)?;

    let mut known_feature = Mat::default();
    recognizer.feature(&known_aligned, &mut known_feature)?;

    println!("✅ Known face feature extracted");

    // Open camera
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?; // 0 = default webcam
    if !cam.is_opened()? {
        panic!("❌ Cannot open camera");
    }

    let cam_width = cam.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let cam_height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

    detector.set_input_size(Size::new(cam_width, cam_height))?;

    println!("📹 Running face recognition on live video...");
    println!("Press 'q' to quit.");

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        if frame.empty() {
            continue;
        }

        let mut faces = Mat::default();
        detector.detect(&frame, &mut faces)?;

        let mut result = frame.try_clone()?;

        if faces.rows() > 0 {
            let mut aligned = Mat::default();
            recognizer.align_crop(&frame, &faces.row(0)?, &mut aligned)?;

            let mut feature = Mat::default();
            recognizer.feature(&aligned, &mut feature)?;

            let cos_sim = recognizer.match_(
                &known_feature,
                &feature,
                FaceRecognizerSF_DisType::FR_COSINE as i32,
            )?;

            // Threshold: >= 0.363
            let text = if cos_sim >= 0.363 {
                "✅ Identity Match!"
            } else {
                "❌ Unknown Person"
            };

            let rect = Rect2f::new(
                *faces.at_2d::<f32>(0, 0)?,
                *faces.at_2d::<f32>(0, 1)?,
                *faces.at_2d::<f32>(0, 2)?,
                *faces.at_2d::<f32>(0, 3)?,
            ).to::<i32>().unwrap();

            imgproc::rectangle(
                &mut result,
                rect,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;

            imgproc::put_text(
                &mut result,
                text,
                rect.tl(),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.8,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                false,
            )?;

            println!("Cosine similarity: {:.3} -> {}", cos_sim, text);
        }

        highgui::imshow("Camera - Face Recognition", &result)?;
        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 {
            break;
        }
    }

    Ok(())
}
