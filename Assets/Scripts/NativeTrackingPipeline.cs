using System;
using System.Reflection;
using UnityEngine;

public class NativeTrackingPipeline : MonoBehaviour
{
    public enum Mode
    {
        TestPlacement,
        RealTracking
    }

    [Header("Mode")]
    public Mode currentMode = Mode.TestPlacement;

    [Header("Scene")]
    public GameObject trackedObject;

    [Header("Passthrough")]
    public GameObject passthroughAccessObject;

    [Header("Placement")]
    public float distanceInFront = 2.5f;

    private Component _cameraAccessComponent;
    private MethodInfo _getCameraPoseMethod;
    private PropertyInfo _isPlayingProperty;

    void Start()
    {
        if (passthroughAccessObject == null) return;

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
                break;
            }
        }
    }

    void Update()
    {
        if (_cameraAccessComponent == null || trackedObject == null)
            return;

        bool isPlaying = (bool)_isPlayingProperty.GetValue(_cameraAccessComponent);
        if (!isPlaying) return;

        Pose camPose = (Pose)_getCameraPoseMethod.Invoke(_cameraAccessComponent, null);

        // 🔹 MODE TEST
        if (currentMode == Mode.TestPlacement)
        {
            trackedObject.transform.position =
                camPose.position + camPose.forward * distanceInFront;

            trackedObject.transform.rotation =
                Quaternion.LookRotation(camPose.forward, Vector3.up);
        }

        // 🔹 MODE TRACKING (vide pour l’instant)
        if (currentMode == Mode.RealTracking)
        {
            // TODO : ici on mettra YOLO plus tard
        }
    }
}