using UnityEngine;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;

public class DetectionReceiver : MonoBehaviour
{
    [SerializeField] private VenopunctureVisualization visualizer;
    [SerializeField] private int listenPort = 5000;
    [SerializeField] private bool autoStart = true;

    private UdpClient listener;
    private Thread receiveThread;
    private bool running = false;

    void Start()
    {
        if (visualizer == null)
            visualizer = GetComponent<VenopunctureVisualization>();

        if (autoStart)
            StartListening();
    }

    public void StartListening()
    {
        if (running) return;

        running = true;
        listener = new UdpClient(listenPort);
        receiveThread = new Thread(ReceiveLoop);
        receiveThread.IsBackground = true;
        receiveThread.Start();
        Debug.Log($"Listening for detection data on port {listenPort}");
    }

    void ReceiveLoop()
    {
        IPEndPoint endPoint = new IPEndPoint(IPAddress.Any, listenPort);

        while (running)
        {
            try
            {
                byte[] data = listener.Receive(ref endPoint);
                string json = System.Text.Encoding.UTF8.GetString(data);
                
                MainThreadDispatcher.Execute(() =>
                {
                    if (visualizer != null)
                        visualizer.LoadFromJSON(json);
                });
            }
            catch (SocketException)
            {
                if (running)
                    Debug.LogWarning("Socket error, reconnecting...");
            }
        }
    }

    public void StopListening()
    {
        running = false;
        if (listener != null)
            listener.Close();
        if (receiveThread != null && receiveThread.IsAlive)
            receiveThread.Join(1000);
    }

    void OnDestroy()
    {
        StopListening();
    }
}

public class MainThreadDispatcher : MonoBehaviour
{
    private static MainThreadDispatcher instance;
    private Queue<System.Action> actions = new Queue<System.Action>();
    private readonly object actionsLock = new object();

    void Awake()
    {
        if (instance == null)
            instance = this;
    }

    void Update()
    {
        while (true)
        {
            System.Action action = null;
            lock (actionsLock)
            {
                if (actions.Count == 0)
                    break;
                action = actions.Dequeue();
            }

            action?.Invoke();
        }
    }

    public static void Execute(System.Action action)
    {
        if (instance == null || action == null)
            return;

        lock (instance.actionsLock)
        {
            instance.actions.Enqueue(action);
        }
    }
}
