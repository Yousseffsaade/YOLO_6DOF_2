using UnityEngine;

public class WebcamFeed : MonoBehaviour
{
    public Renderer display;
    private WebCamTexture webcamTexture;

    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;

        if (devices.Length == 0)
        {
            Debug.LogError("[WebcamFeed] No camera found");
            return;
        }

        for (int i = 0; i < devices.Length; i++)
            Debug.Log($"[WebcamFeed] Device[{i}]: name={devices[i].name} front={devices[i].isFrontFacing}");

        // Pick first non-front-facing camera (rear/passthrough camera)
        int chosen = 0;
        for (int i = 0; i < devices.Length; i++)
        {
            if (!devices[i].isFrontFacing) { chosen = i; break; }
        }

        webcamTexture = new WebCamTexture(devices[chosen].name, 640, 480, 30);

        if (display != null)
        {
            display.material.mainTexture = webcamTexture;
        }

        webcamTexture.Play();

        Debug.Log($"[WebcamFeed] Webcam started: [{chosen}] {devices[chosen].name}");
    }

    public WebCamTexture GetTexture()
    {
        return webcamTexture;
    }
}