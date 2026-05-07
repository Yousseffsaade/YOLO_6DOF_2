using UnityEngine;
using Unity.InferenceEngine;
using System.Reflection;
using TMPro;

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.Calib3dModule;

public class YoloUnityDetector : MonoBehaviour
{
    public ModelAsset modelAsset;
    public WebcamFeed webcamFeed;
    public Transform trackedObject;

    [Header("Passthrough Camera")]
    public GameObject passthroughAccessObject;

    [Header("Debug Display")]
    public TextMeshProUGUI debugText;
    public bool showDebugOnQuest = true;
    public bool verboseConsoleLog = true;

    [Header("Tracking Settings")]
    public float smoothingSpeed = 15f;
    public float trackingScale = 5f;
    public float confidenceThreshold = 0.3f;
    public float fixedDepth = 1.2f;

    [Header("BBox Smoothing")]
    [Range(0.01f, 1f)] public float bboxSmoothing = 0.20f;

    [Header("Movement Limits")]
    public float maxHorizontalOffset = 2f;
    public float maxVerticalOffset = 1.5f;
    public float minObjectY = 0.6f;
    public float maxObjectY = 2.2f;

    [Header("Detection Filtering")]
    public float minBoxWidth = 20f;
    public float minBoxHeight = 40f;
    public float maxBoxWidth = 500f;
    public float maxBoxHeight = 600f;

    [Header("Jump Filtering")]
    public float maxBboxJump = 180f;
    public float maxWorldJump = 1.2f;

    [Header("Lost Detection Handling")]
    public int maxLostFrames = 20;
    public bool hideObjectWhenLost = false;

    [Header("OpenCV SolvePnP")]
    public bool useSolvePnP = true;

    [Tooltip("Approximate real bottle width in meters.")]
    public float bottleWidthMeters = 0.07f;

    [Tooltip("Approximate real bottle height in meters.")]
    public float bottleHeightMeters = 0.22f;

    [Tooltip("Approximate real bottle depth/thickness in meters.")]
    public float bottleDepthMeters = 0.07f;

    [Tooltip("Estimated camera focal length in pixels. Increase if object appears too close/unstable.")]
    public float focalLengthPixels = 650f;

    [Tooltip("If SolvePnP gives an invalid result, fallback to the old fixed-depth method.")]
    public bool fallbackToFixedDepth = true;

    [Tooltip("Minimum accepted SolvePnP distance from camera.")]
    public float minPnPDepth = 0.25f;

    [Tooltip("Maximum accepted SolvePnP distance from camera.")]
    public float maxPnPDepth = 3.0f;

    private Worker worker;
    private WebCamTexture webcamTexture;
    private RenderTexture _resizeRT;
    private Tensor<float> _inputTensor;

    private Vector3 smoothPos = Vector3.zero;
    private Quaternion smoothRot = Quaternion.identity;
    private Vector3 lastValidWorldPos = Vector3.zero;

    private bool smoothPosInitialised = false;
    private bool smoothRotInitialised = false;
    private bool hasValidDetection = false;

    private float smoothX;
    private float smoothY;
    private float smoothW;
    private float smoothH;

    private float lastRawX;
    private float lastRawY;

    private bool smoothBboxInitialised = false;

    private int lostFrames = 0;

    private Component _cameraAccessComponent;
    private MethodInfo _getCameraPoseMethod;
    private PropertyInfo _isPlayingProperty;
    private PropertyInfo _intrinsicsProperty;

    private const int InputSize = 640;
    private const int Predictions = 8400;

    void Start()
    {
        SetDebugText("YOLO DEBUG\nStarting...");

        if (modelAsset == null)
        {
            Debug.LogError("[YOLO] Model Asset is missing.");
            SetDebugText("YOLO ERROR\nModel Asset is missing.");
            return;
        }

        if (webcamFeed == null)
        {
            Debug.LogError("[YOLO] Webcam Feed is missing.");
            SetDebugText("YOLO ERROR\nWebcam Feed is missing.");
            return;
        }

        var model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.GPUCompute);

        webcamTexture = webcamFeed.GetTexture();

        _resizeRT = new RenderTexture(InputSize, InputSize, 0, RenderTextureFormat.ARGB32);
        _resizeRT.Create();
        _inputTensor = new Tensor<float>(new TensorShape(1, 3, InputSize, InputSize));

        Debug.Log("[YOLO] Model loaded");
        SetDebugText("YOLO DEBUG\nModel loaded.\nWaiting for camera...");

        if (passthroughAccessObject != null)
        {
            foreach (var comp in passthroughAccessObject.GetComponents<Component>())
            {
                if (comp == null) continue;

                var t = comp.GetType();
                var poseMethod = t.GetMethod("GetCameraPose");
                var playingProp = t.GetProperty("IsPlaying");

                if (poseMethod != null && playingProp != null)
                {
                    _cameraAccessComponent = comp;
                    _getCameraPoseMethod = poseMethod;
                    _isPlayingProperty = playingProp;
                    _intrinsicsProperty = t.GetProperty("Intrinsics");

                    Debug.Log("[YOLO] Camera pose accessor found");
                    SetDebugText("YOLO DEBUG\nCamera pose accessor found.\nWaiting for detection...");
                    break;
                }
            }
        }

        if (_cameraAccessComponent == null)
        {
            Debug.LogWarning("[YOLO] No camera pose accessor found.");
            SetDebugText("YOLO WARNING\nNo camera pose accessor found.\nWorld position may be wrong.");
        }
    }

    void Update()
    {
        if (worker == null)
        {
            SetDebugText("YOLO ERROR\nWorker is null.");
            return;
        }

        if (webcamTexture == null)
        {
            SetDebugText("YOLO ERROR\nWebcamTexture is null.");
            return;
        }

        if (!webcamTexture.isPlaying)
        {
            SetDebugText("YOLO DEBUG\nWebcamTexture is not playing.");
            return;
        }

        if (!webcamTexture.didUpdateThisFrame)
        {
            return;
        }

        Pose camPose = Pose.identity;

        if (_cameraAccessComponent != null)
        {
            bool isPlaying = (bool)_isPlayingProperty.GetValue(_cameraAccessComponent);

            if (isPlaying)
            {
                camPose = (Pose)_getCameraPoseMethod.Invoke(_cameraAccessComponent, null);
            }
            else
            {
                SetDebugText("YOLO DEBUG\nPassthrough not playing - using identity pose.");
            }

            if (_intrinsicsProperty != null)
            {
                var intrinsics = _intrinsicsProperty.GetValue(_cameraAccessComponent);
                if (intrinsics != null)
                {
                    var focalLengthField = intrinsics.GetType().GetProperty("FocalLength");
                    if (focalLengthField != null)
                    {
                        var fl = (UnityEngine.Vector2)focalLengthField.GetValue(intrinsics);
                        if (fl.x > 0f)
                            focalLengthPixels = fl.x;
                    }
                }
            }
        }

        Graphics.Blit(webcamTexture, _resizeRT);

        if (verboseConsoleLog && Time.frameCount % 90 == 0)
        {
            var sample = webcamTexture.GetPixel(webcamTexture.width / 2, webcamTexture.height / 2);
            Debug.Log($"[YOLO] webcam center pixel: r={sample.r:F2} g={sample.g:F2} b={sample.b:F2} size={webcamTexture.width}x{webcamTexture.height}");
        }

        TextureConverter.ToTensor(_resizeRT, _inputTensor, new TextureTransform());
        worker.Schedule(_inputTensor);

        Tensor<float> outputFloat = worker.PeekOutput() as Tensor<float>;

        if (outputFloat == null)
        {
            SetDebugText("YOLO ERROR\nOutput tensor is not Tensor<float>.");
            return;
        }

        Tensor<float> readableOutput = outputFloat.ReadbackAndClone();
        float[] data = readableOutput.DownloadToArray();

        float bestConf = 0f;
        int bestIndex = -1;

        for (int i = 0; i < Predictions; i++)
        {
            float conf = data[4 * Predictions + i];

            if (conf > bestConf)
            {
                bestConf = conf;
                bestIndex = i;
            }
        }

        if (bestIndex == -1 || bestConf <= confidenceThreshold)
        {
            HandleLostDetection("No detection above confidence threshold.", bestConf);
            readableOutput.Dispose();
            return;
        }

        float x = data[0 * Predictions + bestIndex];
        float y = data[1 * Predictions + bestIndex];
        float w = data[2 * Predictions + bestIndex];
        float h = data[3 * Predictions + bestIndex];

        bool validBoxSize =
            w >= minBoxWidth &&
            h >= minBoxHeight &&
            w <= maxBoxWidth &&
            h <= maxBoxHeight;

        if (!validBoxSize)
        {
            HandleLostDetection(
                $"Rejected bbox size.\nw={w:F1}, h={h:F1}",
                bestConf
            );

            readableOutput.Dispose();
            return;
        }

        if (hasValidDetection && lostFrames <= maxLostFrames)
        {
            float rawJump = Vector2.Distance(
                new Vector2(x, y),
                new Vector2(lastRawX, lastRawY)
            );

            if (rawJump > maxBboxJump)
            {
                HandleLostDetection(
                    $"Rejected bbox jump.\nJump={rawJump:F1}",
                    bestConf
                );

                readableOutput.Dispose();
                return;
            }
        }

        if (!smoothBboxInitialised || lostFrames > maxLostFrames)
        {
            smoothX = x;
            smoothY = y;
            smoothW = w;
            smoothH = h;
            smoothBboxInitialised = true;
        }
        else
        {
            smoothX = Mathf.Lerp(smoothX, x, bboxSmoothing);
            smoothY = Mathf.Lerp(smoothY, y, bboxSmoothing);
            smoothW = Mathf.Lerp(smoothW, w, bboxSmoothing);
            smoothH = Mathf.Lerp(smoothH, h, bboxSmoothing);
        }

        Vector3 worldPos;
        Quaternion worldRot = Quaternion.identity;
        bool pnpOK = false;
        Vector3 pnpCameraPosition = Vector3.zero;
        Quaternion pnpCameraRotation = Quaternion.identity;

        if (useSolvePnP)
        {
            pnpOK = TrySolvePnP(
                smoothX,
                smoothY,
                smoothW,
                smoothH,
                InputSize,
                InputSize,
                out pnpCameraPosition,
                out pnpCameraRotation
            );

            if (pnpOK)
            {
                worldPos =
                    camPose.position +
                    camPose.right * pnpCameraPosition.x +
                    camPose.up * pnpCameraPosition.y +
                    camPose.forward * pnpCameraPosition.z;

                worldRot = camPose.rotation * pnpCameraRotation;
            }
            else if (fallbackToFixedDepth)
            {
                worldPos = GetFixedDepthWorldPosition(camPose, smoothX, smoothY);
            }
            else
            {
                HandleLostDetection("SolvePnP failed and fallback is disabled.", bestConf);

                readableOutput.Dispose();
                return;
            }
        }
        else
        {
            worldPos = GetFixedDepthWorldPosition(camPose, smoothX, smoothY);
        }

        worldPos.y = Mathf.Clamp(worldPos.y, minObjectY, maxObjectY);

        if (hasValidDetection && lostFrames <= maxLostFrames)
        {
            float worldJump = Vector3.Distance(worldPos, lastValidWorldPos);

            if (worldJump > maxWorldJump)
            {
                HandleLostDetection(
                    $"Rejected world jump.\nJump={worldJump:F2} m",
                    bestConf
                );

                readableOutput.Dispose();
                return;
            }
        }

        bool wasLostLong = lostFrames > maxLostFrames;
        lostFrames = 0;
        hasValidDetection = true;
        lastRawX = x;
        lastRawY = y;
        lastValidWorldPos = worldPos;

        if (!smoothPosInitialised || wasLostLong)
        {
            smoothPos = worldPos;
            smoothPosInitialised = true;
        }

        if (!smoothRotInitialised || wasLostLong)
        {
            smoothRot = worldRot;
            smoothRotInitialised = true;
        }

        smoothPos = Vector3.Lerp(
            smoothPos,
            worldPos,
            Time.deltaTime * smoothingSpeed
        );

        smoothPos.y = Mathf.Clamp(smoothPos.y, minObjectY, maxObjectY);

        if (pnpOK)
            smoothRot = Quaternion.Slerp(smoothRot, worldRot, Time.deltaTime * smoothingSpeed);

        if (trackedObject != null)
        {
            trackedObject.gameObject.SetActive(true);
            trackedObject.position = smoothPos;
            if (pnpOK)
                trackedObject.rotation = smoothRot;
        }

        float normX = (smoothX / InputSize) - 0.5f;
        float normY = -((smoothY / InputSize) - 0.5f);

        string debugMessage =
            "YOLO + OPENCV SOLVEPNP\n\n" +

            $"Confidence: {bestConf:F2}\n" +
            $"Threshold: {confidenceThreshold:F2}\n\n" +

            "Raw bbox center:\n" +
            $"x = {x:F1}\n" +
            $"y = {y:F1}\n\n" +

            "Smoothed bbox:\n" +
            $"x = {smoothX:F1}\n" +
            $"y = {smoothY:F1}\n" +
            $"w = {smoothW:F1}\n" +
            $"h = {smoothH:F1}\n\n" +

            "Normalized position:\n" +
            $"x = {normX:F2}\n" +
            $"y = {normY:F2}\n\n" +

            "SolvePnP:\n" +
            $"enabled = {useSolvePnP}\n" +
            $"status = {(pnpOK ? "OK" : "FAIL / FALLBACK")}\n" +
            $"camera x = {pnpCameraPosition.x:F2}\n" +
            $"camera y = {pnpCameraPosition.y:F2}\n" +
            $"camera z = {pnpCameraPosition.z:F2}\n\n" +

            "Camera pose:\n" +
            $"x = {camPose.position.x:F2}\n" +
            $"y = {camPose.position.y:F2}\n" +
            $"z = {camPose.position.z:F2}\n\n" +

            "World target position:\n" +
            $"x = {worldPos.x:F2}\n" +
            $"y = {worldPos.y:F2}\n" +
            $"z = {worldPos.z:F2}\n\n" +

            "Object smooth position:\n" +
            $"x = {smoothPos.x:F2}\n" +
            $"y = {smoothPos.y:F2}\n" +
            $"z = {smoothPos.z:F2}\n\n" +

            $"Lost frames: {lostFrames}";

        SetDebugText(debugMessage);

        if (verboseConsoleLog)
            Debug.Log(
                $"[YOLO] conf={bestConf:F2} " +
                $"raw=({x:F1},{y:F1}) smooth=({smoothX:F1},{smoothY:F1}) " +
                $"pnp={(pnpOK ? "OK" : "FAIL")} " +
                $"cam=({pnpCameraPosition.x:F2},{pnpCameraPosition.y:F2},{pnpCameraPosition.z:F2}) " +
                $"world=({worldPos.x:F2},{worldPos.y:F2},{worldPos.z:F2})"
            );

        readableOutput.Dispose();
    }

    Vector3 GetFixedDepthWorldPosition(Pose camPose, float bboxCenterX, float bboxCenterY)
    {
        float normX = (bboxCenterX / InputSize) - 0.5f;
        float normY = -((bboxCenterY / InputSize) - 0.5f);

        float horizontalOffset = Mathf.Clamp(
            normX * trackingScale,
            -maxHorizontalOffset,
            maxHorizontalOffset
        );

        float verticalOffset = Mathf.Clamp(
            normY * trackingScale,
            -maxVerticalOffset,
            maxVerticalOffset
        );

        return
            camPose.position +
            camPose.forward * fixedDepth +
            camPose.right * horizontalOffset +
            camPose.up * verticalOffset;
    }

    bool TrySolvePnP(
        float bboxCenterX,
        float bboxCenterY,
        float bboxWidth,
        float bboxHeight,
        int imageWidth,
        int imageHeight,
        out Vector3 cameraSpacePosition,
        out Quaternion cameraSpaceRotation
    )
    {
        cameraSpacePosition = Vector3.zero;
        cameraSpaceRotation = Quaternion.identity;

        float halfW = bboxWidth / 2f;
        float halfH = bboxHeight / 2f;

        float left = bboxCenterX - halfW;
        float right = bboxCenterX + halfW;
        float top = bboxCenterY - halfH;
        float bottom = bboxCenterY + halfH;

        MatOfPoint3f objectPoints = new MatOfPoint3f(
            new Point3(-bottleWidthMeters / 2.0,  bottleHeightMeters / 2.0, 0),
            new Point3( bottleWidthMeters / 2.0,  bottleHeightMeters / 2.0, 0),
            new Point3( bottleWidthMeters / 2.0, -bottleHeightMeters / 2.0, 0),
            new Point3(-bottleWidthMeters / 2.0, -bottleHeightMeters / 2.0, 0)
        );

        MatOfPoint2f imagePoints = new MatOfPoint2f(
            new Point(left, top),
            new Point(right, top),
            new Point(right, bottom),
            new Point(left, bottom)
        );

        Mat cameraMatrix = Mat.eye(3, 3, CvType.CV_64FC1);
        cameraMatrix.put(0, 0, focalLengthPixels);
        cameraMatrix.put(0, 1, 0);
        cameraMatrix.put(0, 2, imageWidth / 2.0);

        cameraMatrix.put(1, 0, 0);
        cameraMatrix.put(1, 1, focalLengthPixels);
        cameraMatrix.put(1, 2, imageHeight / 2.0);

        cameraMatrix.put(2, 0, 0);
        cameraMatrix.put(2, 1, 0);
        cameraMatrix.put(2, 2, 1);

        MatOfDouble distCoeffs = new MatOfDouble(0, 0, 0, 0);

        Mat rvec = new Mat();
        Mat tvec = new Mat();

        bool success = false;

        try
        {
            success = Calib3d.solvePnP(
                objectPoints,
                imagePoints,
                cameraMatrix,
                distCoeffs,
                rvec,
                tvec,
                false,
                Calib3d.SOLVEPNP_ITERATIVE
            );

            if (!success)
            {
                ReleasePnPMats(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
                return false;
            }

            double[] t = new double[3];
            tvec.get(0, 0, t);

            float x = (float)t[0];
            float y = -(float)t[1];
            float z = (float)t[2];

            if (float.IsNaN(x) || float.IsNaN(y) || float.IsNaN(z))
            {
                ReleasePnPMats(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
                return false;
            }

            if (z < minPnPDepth || z > maxPnPDepth)
            {
                ReleasePnPMats(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
                return false;
            }

            cameraSpacePosition = new Vector3(x, y, z);
            cameraSpaceRotation = RvecToUnityQuaternion(rvec);

            ReleasePnPMats(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
            return true;
        }
        catch (System.Exception e)
        {
            Debug.LogWarning("[YOLO OPENCV] SolvePnP exception: " + e.Message);
            ReleasePnPMats(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
            return false;
        }
    }

    Quaternion RvecToUnityQuaternion(Mat rvec)
    {
        Mat rotMat = new Mat();
        Calib3d.Rodrigues(rvec, rotMat);

        double[] r = new double[9];
        rotMat.get(0, 0, r);
        rotMat.Dispose();

        // Convert OpenCV rotation (right-handed, Y-down) to Unity (left-handed, Y-up)
        // by applying C * R * C where C = diag(1,-1,1), which negates all elements
        // where exactly one index equals 1 (the Y row/column).
        float r00 = (float) r[0]; float r01 = -(float)r[1]; float r02 = (float) r[2];
        float r10 = -(float)r[3]; float r11 = (float) r[4]; float r12 = -(float)r[5];
        float r20 = (float) r[6]; float r21 = -(float)r[7]; float r22 = (float) r[8];

        // Unity Matrix4x4 is column-major
        Matrix4x4 m = Matrix4x4.identity;
        m.SetColumn(0, new Vector4(r00, r10, r20, 0));
        m.SetColumn(1, new Vector4(r01, r11, r21, 0));
        m.SetColumn(2, new Vector4(r02, r12, r22, 0));

        return m.rotation;
    }

    void ReleasePnPMats(
        MatOfPoint3f objectPoints,
        MatOfPoint2f imagePoints,
        Mat cameraMatrix,
        MatOfDouble distCoeffs,
        Mat rvec,
        Mat tvec
    )
    {
        objectPoints?.Dispose();
        imagePoints?.Dispose();
        cameraMatrix?.Dispose();
        distCoeffs?.Dispose();
        rvec?.Dispose();
        tvec?.Dispose();
    }

    void HandleLostDetection(string reason, float bestConf)
    {
        lostFrames++;

        if (trackedObject != null)
        {
            if (hideObjectWhenLost && lostFrames > maxLostFrames)
            {
                trackedObject.gameObject.SetActive(false);
            }
            else if (hasValidDetection)
            {
                trackedObject.gameObject.SetActive(true);
                trackedObject.position = smoothPos;
            }
        }

        string message =
            "YOLO FILTER\n\n" +
            "Detection rejected / lost.\n\n" +
            $"Reason:\n{reason}\n\n" +
            $"Best confidence: {bestConf:F2}\n" +
            $"Threshold: {confidenceThreshold:F2}\n\n" +
            $"Lost frames: {lostFrames}\n" +
            $"Max lost frames: {maxLostFrames}\n\n" +
            "Object behavior:\n" +
            (hideObjectWhenLost
                ? "Will hide if lost too long."
                : "Holding last valid position.");

        SetDebugText(message);

        if (verboseConsoleLog)
            Debug.LogWarning($"[YOLO] Lost: {reason} conf={bestConf:F2} lost={lostFrames}");
    }

    void SetDebugText(string message)
    {
        if (!showDebugOnQuest) return;

        if (debugText != null)
        {
            debugText.text = message;
        }
    }

    void OnDestroy()
    {
        worker?.Dispose();
        _inputTensor?.Dispose();
        if (_resizeRT != null) { _resizeRT.Release(); _resizeRT = null; }
    }
}