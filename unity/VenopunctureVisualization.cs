using UnityEngine;
using System.Collections.Generic;

[System.Serializable]
public class DetectionFrame
{
    public int frame;
    public string timestamp;
    public BoardData board;
    public ArmData arm;
    public VeinData vein;
}

[System.Serializable]
public class BoardData
{
    public bool detected;
    public float[][] corners;
    public int[] ids;
    public PoseData pose;
}

[System.Serializable]
public class ArmData
{
    public bool detected;
    public BboxData bbox;
}

[System.Serializable]
public class BboxData
{
    public float x, y, width, height;
}

[System.Serializable]
public class PoseData
{
    public float[] rvec;
    public float[] tvec;
    public float distance_m;
}

[System.Serializable]
public class VeinData
{
    public bool detected;
    public float[] contour;
    public float confidence;
}

[System.Serializable]
public class DetectionData
{
    public DetectionFrame[] detections;
}

[System.Serializable]
public class DetectionEnvelope
{
    public DetectionFrame[] detections;
}

public class VenopunctureVisualization : MonoBehaviour
{
    [SerializeField] private GameObject boardQuad;
    [SerializeField] private GameObject armQuad;
    [SerializeField] private float boardScale = 0.25f;
    [SerializeField] private float boardHeight = 0.2f;
    [SerializeField] private float axisLength = 0.1f;
    [SerializeField] private bool drawAxes = true;
    [SerializeField] private float armSurfaceOffset = 0.004f;
    [SerializeField] private float armMaxHeightOnBoard = 0.25f;
    [SerializeField] private float armMinHeightOnBoard = -0.35f;
    [SerializeField] private int inputImageWidth = 1280;
    [SerializeField] private int inputImageHeight = 720;
    [SerializeField] private float veinLineWidth = 0.003f;
    [SerializeField] private Color veinColor = new Color(0.0f, 0.6f, 1.0f, 0.9f);

    private List<DetectionFrame> frames = new List<DetectionFrame>();
    private int currentFrame = 0;
    private bool loaded = false;
    private Renderer armRenderer;
    private Color armDetectedColor = new Color(0.45f, 0.45f, 0.45f, 0.85f);
    private Color armUndetectedColorA = new Color(0.9f, 0.8f, 0.1f, 0.35f);
    private Color armUndetectedColorB = new Color(1.0f, 0.45f, 0.1f, 0.55f);
    private LineRenderer veinLineRenderer;
    private float boardAspectRatio = 0.714f;

    void Start()
    {
        CreateQuads();
        CreateVeinRenderer();
        LoadBoardTexture();
    }

    void Update()
    {
        if (!loaded)
        {
            boardQuad.SetActive(true);
            ShowArmPlaceholder();
            SetArmUndetectedVisual();
            return;
        }

        if (Input.GetKeyDown(KeyCode.RightArrow))
            currentFrame = (currentFrame + 1) % frames.Count;
        if (Input.GetKeyDown(KeyCode.LeftArrow))
            currentFrame = (currentFrame - 1 + frames.Count) % frames.Count;

        UpdateFrame();
    }

    public void LoadFromJSON(string jsonText)
    {
        try
        {
            DetectionData data = JsonUtility.FromJson<DetectionData>(jsonText);
            if ((data == null || data.detections == null || data.detections.Length == 0) && jsonText.Contains("\"detections\""))
            {
                DetectionEnvelope envelope = JsonUtility.FromJson<DetectionEnvelope>(jsonText);
                if (envelope != null)
                {
                    data = new DetectionData { detections = envelope.detections };
                }
            }

            if (data == null || data.detections == null || data.detections.Length == 0)
            {
                Debug.LogWarning("No detections found in JSON payload.");
                return;
            }

            frames.Clear();
            frames.AddRange(data.detections);
            loaded = true;
            Debug.Log($"Loaded {frames.Count} frames");
        }
        catch (System.Exception e)
        {
            Debug.LogError("JSON parse error: " + e.Message);
        }
    }

    public void LoadFromFile(string path)
    {
        if (System.IO.File.Exists(path))
        {
            string json = System.IO.File.ReadAllText(path);
            LoadFromJSON(json);
        }
    }

    void CreateQuads()
    {
        if (boardQuad == null)
        {
            boardQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            boardQuad.name = "Board";
            boardQuad.transform.parent = transform;
            boardQuad.transform.localPosition = new Vector3(0f, boardHeight, 0.5f);
            boardQuad.transform.localScale = new Vector3(boardScale, boardScale / boardAspectRatio, 1f);
            Destroy(boardQuad.GetComponent<Collider>());

            var renderer = boardQuad.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material.color = Color.white;
            }
        }

        if (armQuad == null)
        {
            armQuad = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            armQuad.name = "Arm";
            armQuad.transform.parent = transform;
            armQuad.SetActive(false);
            Destroy(armQuad.GetComponent<Collider>());

            var renderer = armQuad.GetComponent<Renderer>();
            if (renderer != null)
            {
                armRenderer = renderer;
                armRenderer.material.color = armDetectedColor;
            }
        }
        else
        {
            armRenderer = armQuad.GetComponent<Renderer>();
        }
    }

    void CreateVeinRenderer()
    {
        GameObject veinObj = new GameObject("VeinLines");
        veinObj.transform.parent = transform;
        veinObj.transform.localPosition = Vector3.zero;
        veinObj.transform.localRotation = Quaternion.identity;

        veinLineRenderer = veinObj.AddComponent<LineRenderer>();

        Material lineMat = new Material(Shader.Find("Sprites/Default"));
        lineMat.color = veinColor;
        veinLineRenderer.material = lineMat;
        veinLineRenderer.startColor = veinColor;
        veinLineRenderer.endColor = veinColor;
        veinLineRenderer.startWidth = veinLineWidth;
        veinLineRenderer.endWidth = veinLineWidth;
        veinLineRenderer.positionCount = 0;
        veinLineRenderer.enabled = false;
    }

    void LoadBoardTexture()
    {
        string path = Application.dataPath + "/Scripts/Venipuncture/charuco_board.png";
        if (System.IO.File.Exists(path))
        {
            byte[] bytes = System.IO.File.ReadAllBytes(path);
            Texture2D tex = new Texture2D(2, 2, TextureFormat.RGBA32, true);
            tex.LoadImage(bytes);
            tex.filterMode = FilterMode.Trilinear;
            tex.anisoLevel = 8;
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.Apply(true, false);
            boardQuad.GetComponent<Renderer>().material.mainTexture = tex;
            boardQuad.GetComponent<Renderer>().material.color = Color.white;
            boardAspectRatio = (float)tex.width / tex.height;
            Debug.Log($"Loaded board texture: {tex.width}x{tex.height} aspect={boardAspectRatio:F3}");
        }
        else
        {
            Debug.LogWarning($"Board texture not found at: {path}");
        }
    }

    void UpdateFrame()
    {
        if (currentFrame >= frames.Count) return;

        DetectionFrame f = frames[currentFrame];

        if (f.board != null && f.board.detected)
        {
            boardQuad.SetActive(true);
            if (f.board.pose != null && f.board.pose.tvec != null && f.board.pose.tvec.Length >= 3 && f.board.pose.rvec != null && f.board.pose.rvec.Length >= 3)
                UpdateBoardPose(f.board.pose);
            else
                ShowBoardDefault();
        }
        else
        {
            ShowBoardDefault();
        }

        if (f.arm != null && f.arm.detected)
        {
            if (f.arm.bbox != null)
                ShowArmBbox(f.arm.bbox);
            else
                ShowArmPlaceholder();

            SetArmDetectedVisual();
        }
        else
        {
            ShowArmPlaceholder();
            SetArmUndetectedVisual();
        }

        if (f.vein != null && f.vein.detected && f.vein.contour != null && f.vein.contour.Length >= 4)
            ShowVeins(f.vein.contour);
        else
            veinLineRenderer.enabled = false;
    }

    void ShowBoardDefault()
    {
        boardQuad.transform.localPosition = new Vector3(0f, boardHeight, 0.5f);
        boardQuad.transform.localRotation = Quaternion.identity;
        boardQuad.transform.localScale = new Vector3(boardScale, boardScale / boardAspectRatio, 1f);
    }

    void UpdateBoardPose(PoseData pose)
    {
        float angle = new Vector3(pose.rvec[0], pose.rvec[1], pose.rvec[2]).magnitude;
        Vector3 axis = angle > 0.001f ? new Vector3(pose.rvec[0], pose.rvec[1], pose.rvec[2]) / angle : Vector3.up;
        Quaternion rot = Quaternion.AngleAxis(angle * Mathf.Rad2Deg, axis);

        boardQuad.transform.position = new Vector3(pose.tvec[0], boardHeight + pose.tvec[1], pose.tvec[2]);
        boardQuad.transform.rotation = rot;
        boardQuad.transform.localScale = new Vector3(boardScale, boardScale / boardAspectRatio, 1);

        if (drawAxes)
        {
            Debug.DrawLine(boardQuad.transform.position, boardQuad.transform.position + rot * Vector3.right * axisLength, Color.red);
            Debug.DrawLine(boardQuad.transform.position, boardQuad.transform.position + rot * Vector3.up * axisLength, Color.green);
            Debug.DrawLine(boardQuad.transform.position, boardQuad.transform.position + rot * Vector3.forward * axisLength, Color.blue);
        }
    }

    void ShowArmBbox(BboxData bbox)
    {
        armQuad.SetActive(true);

        float boardWidth = boardScale;
        float boardHeightWorld = boardScale / boardAspectRatio;
        float halfW = boardWidth * 0.5f;
        float halfH = boardHeightWorld * 0.5f;

        float centerX = bbox.x + (bbox.width * 0.5f);
        float centerY = bbox.y + (bbox.height * 0.5f);

        float nx = (centerX / Mathf.Max(1, inputImageWidth)) - 0.5f;
        float ny = 0.5f - (centerY / Mathf.Max(1, inputImageHeight));

        float localX = nx * boardWidth;
        float localY = ny * boardHeightWorld;
        localY = Mathf.Clamp(localY, halfH * armMinHeightOnBoard, halfH * armMaxHeightOnBoard);

        float targetW = Mathf.Clamp((bbox.width / Mathf.Max(1, inputImageWidth)) * boardWidth * 1.35f, 0.05f, boardWidth * 0.65f);
        float targetH = Mathf.Clamp((bbox.height / Mathf.Max(1, inputImageHeight)) * boardHeightWorld * 1.35f, 0.03f, boardHeightWorld * 0.65f);

        float halfArmW = targetW * 0.5f;
        float halfArmH = targetH * 0.5f;
        localX = Mathf.Clamp(localX, -halfW + halfArmW, halfW - halfArmW);
        localY = Mathf.Clamp(localY, -halfH + halfArmH, halfH - halfArmH);

        PlaceArmOnBoard(localX, localY, targetW, targetH);
    }

    void ShowArmPlaceholder()
    {
        armQuad.SetActive(true);
        PlaceArmOnBoard(0f, -0.02f, 0.18f, 0.08f);
    }

    void PlaceArmOnBoard(float localX, float localY, float width, float height)
    {
        Vector3 boardPos = boardQuad.activeSelf ? boardQuad.transform.position : transform.position + new Vector3(0f, boardHeight, 0.5f);
        Quaternion boardRot = boardQuad.activeSelf ? boardQuad.transform.rotation : Quaternion.identity;

        Vector3 targetPos = boardPos
            + (boardRot * Vector3.right * localX)
            + (boardRot * Vector3.up * localY)
            + (boardRot * Vector3.forward * armSurfaceOffset);

        armQuad.transform.position = targetPos;

        // Cylinder default: height=2 along Y, radius=1 in X/Z
        // Orient cylinder along the arm's longer axis
        float thickness = Mathf.Min(width, height) * 0.35f;
        if (width >= height)
        {
            armQuad.transform.rotation = boardRot * Quaternion.Euler(0f, 0f, 90f);
            armQuad.transform.localScale = new Vector3(width / 2f, thickness, thickness);
        }
        else
        {
            armQuad.transform.rotation = boardRot;
            armQuad.transform.localScale = new Vector3(thickness, height / 2f, thickness);
        }
    }

    void SetArmDetectedVisual()
    {
        if (armRenderer == null)
            return;

        if (armRenderer.material.HasProperty("_Color"))
            armRenderer.material.color = armDetectedColor;
    }

    void SetArmUndetectedVisual()
    {
        if (armRenderer == null)
            return;

        float t = (Mathf.Sin(Time.time * 2.7f) + 1f) * 0.5f;
        if (armRenderer.material.HasProperty("_Color"))
            armRenderer.material.color = Color.Lerp(armUndetectedColorA, armUndetectedColorB, t);
    }

    void ShowVeins(float[] contour)
    {
        int pointCount = contour.Length / 2;

        float boardWidth = boardScale;
        float boardHeightWorld = boardScale / boardAspectRatio;
        float halfW = boardWidth * 0.5f;
        float halfH = boardHeightWorld * 0.5f;

        Vector3 boardPos = boardQuad.activeSelf ? boardQuad.transform.position : transform.position + new Vector3(0f, boardHeight, 0.5f);
        Quaternion boardRot = boardQuad.activeSelf ? boardQuad.transform.rotation : Quaternion.identity;

        Vector3[] positions = new Vector3[pointCount];
        for (int i = 0; i < pointCount; i++)
        {
            float px = contour[i * 2];
            float py = contour[i * 2 + 1];

            float nx = (px / Mathf.Max(1, inputImageWidth)) - 0.5f;
            float ny = 0.5f - (py / Mathf.Max(1, inputImageHeight));

            float localX = nx * boardWidth;
            float localY = ny * boardHeightWorld;

            localX = Mathf.Clamp(localX, -halfW * 0.9f, halfW * 0.9f);
            localY = Mathf.Clamp(localY, -halfH * 0.9f, halfH * 0.9f);

            Vector3 worldPos = boardPos
                + (boardRot * Vector3.right * localX)
                + (boardRot * Vector3.up * localY)
                + (boardRot * Vector3.forward * (armSurfaceOffset + 0.001f));

            positions[i] = worldPos;
        }

        veinLineRenderer.positionCount = pointCount;
        veinLineRenderer.SetPositions(positions);
        veinLineRenderer.enabled = true;
    }

    public int GetCurrentFrame() => currentFrame;
    public int GetTotalFrames() => frames.Count;
    public bool IsLoaded() => loaded;
}
