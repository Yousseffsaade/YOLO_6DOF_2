using System;
using System.Reflection;
using UnityEngine;

public class CameraFrameReader : MonoBehaviour
{
    public GameObject passthroughAccessObject;

    private Component _cameraAccessComponent;
    private MethodInfo _getCameraPoseMethod;
    private PropertyInfo _isPlayingProperty;

    void Start()
    {
        if (passthroughAccessObject == null)
        {
            Debug.LogError("Assign Passthrough Camera Access!");
            return;
        }

        foreach (var comp in passthroughAccessObject.GetComponents<Component>())
        {
            if (comp == null) continue;

            Type t = comp.GetType();

            var poseMethod = t.GetMethod("GetCameraPose");
            var playingProp = t.GetProperty("IsPlaying");

            if (poseMethod != null && playingProp != null)
            {
                _cameraAccessComponent = comp;
                _getCameraPoseMethod = poseMethod;
                _isPlayingProperty = playingProp;

                Debug.Log("[CameraFrameReader] Camera access FOUND");
                break;
            }
        }

        if (_cameraAccessComponent == null)
        {
            Debug.LogError("[CameraFrameReader] Camera access NOT found");
        }
    }

    void Update()
    {
        if (_cameraAccessComponent == null)
            return;

        bool isPlaying = (bool)_isPlayingProperty.GetValue(_cameraAccessComponent);

        if (!isPlaying)
            return;

        Pose camPose = (Pose)_getCameraPoseMethod.Invoke(_cameraAccessComponent, null);

        // 🔥 TEST : juste afficher dans la console
        Debug.Log("[CameraFrameReader] Camera running | Pos: " + camPose.position);
    }
}