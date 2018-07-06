#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>


#define FORCEINLINE __forceinline
#define ALIGN(a) __declspec(align(a))

#define k_DemoName "Real-time Quaternion Julia Sets (CPU, AVX2)"
#define k_DemoResolutionX 800
#define k_DemoResolutionY 800
#define k_DemoRcpResolutionX (1.0f / k_DemoResolutionX)
#define k_DemoRcpResolutionY (1.0f / k_DemoResolutionY)
#define k_DemoAspectRatio ((float)k_DemoResolutionX / k_DemoResolutionY)

#define k_TileSize 16
#define k_TileCountX (k_DemoResolutionX / k_TileSize)
#define k_TileCountY (k_DemoResolutionY / k_TileSize)
#define k_TileCount (k_TileCountX * k_TileCountY)

typedef __m256 floatpacket;
typedef __m256i intpacket;

typedef ALIGN(32) struct vec3packet
{
    floatpacket X, Y, Z;
} vec3packet;

typedef ALIGN(32) struct vec4packet
{
    floatpacket X, Y, Z, W;
} vec4packet;

typedef struct job_data
{
    uint8_t *DisplayPtr;
} job_data;

typedef struct demo
{
    HWND Window;
    HDC WindowDc;
    HDC MemoryDc;
    uint32_t CpuCoreCount;
    job_data Job;
    PTP_WORK JobHandle;
} demo;

static const floatpacket k_1_0 = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
static const floatpacket k_255_0 = { 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f, 255.0f };

ALIGN(64) static volatile uint32_t s_TileIndex[16];

static FORCEINLINE vec3packet
Vec3PacketNormalize(vec3packet Vector)
{
    floatpacket ReciprocalLength;
    ReciprocalLength = _mm256_mul_ps(Vector.X, Vector.X);
    ReciprocalLength = _mm256_fmadd_ps(Vector.Y, Vector.Y, ReciprocalLength);
    ReciprocalLength = _mm256_fmadd_ps(Vector.Z, Vector.Z, ReciprocalLength);
    ReciprocalLength = _mm256_sqrt_ps(ReciprocalLength);
    ReciprocalLength = _mm256_div_ps(k_1_0, ReciprocalLength);

    vec3packet Result;
    Result.X = _mm256_mul_ps(Vector.X, ReciprocalLength);
    Result.Y = _mm256_mul_ps(Vector.Y, ReciprocalLength);
    Result.Z = _mm256_mul_ps(Vector.Z, ReciprocalLength);

    return Result;
}

static FORCEINLINE vec4packet
QuatMultiply(vec4packet Quat0, vec4packet Quat1)
{
    vec4packet Result;

    Result.W = _mm256_mul_ps(Quat0.W, Quat1.W);
    Result.W = _mm256_fnmadd_ps(Quat0.X, Quat1.X, Result.W);
    Result.W = _mm256_fnmadd_ps(Quat0.Y, Quat1.Y, Result.W);
    Result.W = _mm256_fnmadd_ps(Quat0.Z, Quat1.Z, Result.W);

    Result.X = _mm256_mul_ps(Quat0.W, Quat1.X);
    Result.X = _mm256_fmadd_ps(Quat0.X, Quat1.W, Result.X);
    Result.X = _mm256_fmadd_ps(Quat0.Y, Quat1.Z, Result.X);
    Result.X = _mm256_fnmadd_ps(Quat0.Z, Quat1.Y, Result.X);

    Result.Y = _mm256_mul_ps(Quat0.W, Quat1.Y);
    Result.Y = _mm256_fnmadd_ps(Quat0.X, Quat1.Z, Result.Y);
    Result.Y = _mm256_fmadd_ps(Quat0.Y, Quat1.W, Result.Y);
    Result.Y = _mm256_fmadd_ps(Quat0.Z, Quat1.X, Result.Y);

    Result.Z = _mm256_mul_ps(Quat0.W, Quat1.Z);
    Result.Z = _mm256_fmadd_ps(Quat0.X, Quat1.Y, Result.Z);
    Result.Z = _mm256_fnmadd_ps(Quat0.Y, Quat1.X, Result.Z);
    Result.Z = _mm256_fmadd_ps(Quat0.Z, Quat1.W, Result.Z);

    return Result;
}

static FORCEINLINE vec4packet
QuatSquare(vec4packet Quat)
{
    vec4packet Result;

    Result.W = _mm256_mul_ps(Quat.W, Quat.W);
    Result.W = _mm256_fnmadd_ps(Quat.X, Quat.X, Result.W);
    Result.W = _mm256_fnmadd_ps(Quat.Y, Quat.Y, Result.W);
    Result.W = _mm256_fnmadd_ps(Quat.Z, Quat.Z, Result.W);

    Result.X = _mm256_mul_ps(Quat.X, Quat.W);
    Result.Y = _mm256_mul_ps(Quat.Y, Quat.W);
    Result.Z = _mm256_mul_ps(Quat.Z, Quat.W);

    Result.X = _mm256_add_ps(Result.X, Result.X);
    Result.Y = _mm256_add_ps(Result.Y, Result.Y);
    Result.Z = _mm256_add_ps(Result.Z, Result.Z);

    return Result;
}

static double
GetTime(void)
{
    static LARGE_INTEGER StartCounter;
    static LARGE_INTEGER Frequency;
    if (StartCounter.QuadPart == 0)
    {
        QueryPerformanceFrequency(&Frequency);
        QueryPerformanceCounter(&StartCounter);
    }
    LARGE_INTEGER Counter;
    QueryPerformanceCounter(&Counter);
    return (Counter.QuadPart - StartCounter.QuadPart) / (double)Frequency.QuadPart;
}

static void
UpdateFrameTime(HWND Window, double *Time, float *DeltaTime)
{
    static double PreviousTime = -1.0;
    static double TextRefreshTime = 0.0;
    static uint32_t FrameCount = 0;

    if (PreviousTime < 0.0)
    {
        PreviousTime = GetTime();
        TextRefreshTime = PreviousTime;
    }

    *Time = GetTime();
    *DeltaTime = (float)(*Time - PreviousTime);
    PreviousTime = *Time;
    FrameCount++;

    if ((*Time - TextRefreshTime) >= 1.0)
    {
        double FramesPerSecond = FrameCount / (*Time - TextRefreshTime);
        double Milliseconds = (1.0 / FramesPerSecond) * 1000.0;
        char Text[256];
        snprintf(Text, sizeof(Text), "[%.1f fps  %.3f ms] %s", FramesPerSecond, Milliseconds, k_DemoName);
        SetWindowText(Window, Text);
        TextRefreshTime = *Time;
        FrameCount = 0;
    }
}

static LRESULT CALLBACK
ProcessWindowMessage(HWND Window, UINT Message, WPARAM ParamW, LPARAM ParamL)
{
    switch (Message)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_KEYDOWN:
        if (ParamW == VK_ESCAPE)
        {
            PostQuitMessage(0);
            return 0;
        }
        break;
    }
    return DefWindowProc(Window, Message, ParamW, ParamL);
}

static void
InitializeWindow(demo *Demo)
{
    WNDCLASS Winclass = {
        .lpfnWndProc = ProcessWindowMessage,
        .hInstance = GetModuleHandle(NULL),
        .hCursor = LoadCursor(NULL, IDC_ARROW),
        .lpszClassName = k_DemoName
    };
    if (!RegisterClass(&Winclass))
        assert(0);

    RECT Rect = { 0, 0, k_DemoResolutionX, k_DemoResolutionY };
    if (!AdjustWindowRect(&Rect, WS_OVERLAPPED | WS_SYSMENU | WS_CAPTION | WS_MINIMIZEBOX, 0))
        assert(0);

    Demo->Window = CreateWindowEx(
        0, k_DemoName, k_DemoName, WS_OVERLAPPED | WS_SYSMENU | WS_CAPTION | WS_MINIMIZEBOX | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        Rect.right - Rect.left, Rect.bottom - Rect.top,
        NULL, NULL, NULL, 0);
    assert(Demo->Window);

    Demo->WindowDc = GetDC(Demo->Window);
    assert(Demo->WindowDc);

    BITMAPINFO BitmapInfo = {
        .bmiHeader.biSize = sizeof(BITMAPINFOHEADER),
        .bmiHeader.biPlanes = 1,
        .bmiHeader.biBitCount = 32,
        .bmiHeader.biCompression = BI_RGB,
        .bmiHeader.biWidth = k_DemoResolutionX,
        .bmiHeader.biHeight = k_DemoResolutionY,
        .bmiHeader.biSizeImage = k_DemoResolutionX * k_DemoResolutionY
    };
    HBITMAP BitmapHandle = CreateDIBSection(Demo->WindowDc, &BitmapInfo, DIB_RGB_COLORS, (void **)&Demo->Job.DisplayPtr,
                                            NULL, 0);
    assert(BitmapHandle);

    Demo->MemoryDc = CreateCompatibleDC(Demo->WindowDc);
    assert(Demo->MemoryDc);

    SelectObject(Demo->MemoryDc, BitmapHandle);
}

static void CALLBACK
DrawTiles(PTP_CALLBACK_INSTANCE Instance, void *Context, PTP_WORK Work)
{
    (void)Instance;
    (void)Work;
    job_data *Data = (job_data *)Context;
    uint8_t *DisplayPtr = Data->DisplayPtr;

    for (;;)
    {
        uint32_t TileIndex = (uint32_t)_InterlockedIncrement((volatile LONG *)s_TileIndex) - 1;
        if (TileIndex >= k_TileCount)
            break;

        uint32_t BeginX = (TileIndex % k_TileCountX) * k_TileSize;
        uint32_t BeginY = (TileIndex / k_TileCountX) * k_TileSize;
        uint32_t EndX = BeginX + k_TileSize;
        uint32_t EndY = BeginY + k_TileSize;

        for (uint32_t CurrentY = BeginY; CurrentY < EndY; ++CurrentY)
        {
            for (uint32_t CurrentX = BeginX; CurrentX < EndX; CurrentX += 8)
            {
                float CoordX = (float)CurrentX - k_DemoResolutionX / 2;
                float CoordY = (float)CurrentY - k_DemoResolutionY / 2;

                vec3packet RayDirection;
                RayDirection.X = _mm256_add_ps(_mm256_broadcast_ss(&CoordX),
                                               _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f));
                RayDirection.Y = _mm256_broadcast_ss(&CoordY);
                RayDirection.Z = _mm256_set1_ps(-1.5f);

                RayDirection.X = _mm256_mul_ps(RayDirection.X, _mm256_set1_ps(2.0f / k_DemoResolutionX));
                RayDirection.Y = _mm256_mul_ps(RayDirection.Y, _mm256_set1_ps(2.0f / k_DemoResolutionY));

                RayDirection = Vec3PacketNormalize(RayDirection);

                RayDirection.X = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(RayDirection.X, k_1_0));
                RayDirection.Y = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(RayDirection.Y, k_1_0));
                RayDirection.Z = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(RayDirection.Z, k_1_0));

                intpacket ColorR, ColorG, ColorB;
                ColorR = _mm256_cvttps_epi32(_mm256_mul_ps(RayDirection.X, k_255_0));
                ColorG = _mm256_cvttps_epi32(_mm256_mul_ps(RayDirection.Y, k_255_0));
                ColorB = _mm256_cvttps_epi32(_mm256_mul_ps(RayDirection.Z, k_255_0));

                intpacket Color;
                Color = _mm256_or_si256(ColorB, _mm256_or_si256(_mm256_slli_epi32(ColorR, 16), _mm256_slli_epi32(ColorG, 8)));

                uint32_t Index = (CurrentX + CurrentY * k_DemoResolutionX) * 4;
                _mm256_store_si256((__m256i *)&DisplayPtr[Index], Color);
            }
        }
    }
}

static void
Initialize(demo *Demo)
{
    SYSTEM_INFO SystemInfo;
    GetSystemInfo(&SystemInfo);
    Demo->CpuCoreCount = (uint32_t)SystemInfo.dwNumberOfProcessors;

    Demo->JobHandle = CreateThreadpoolWork(DrawTiles, &Demo->Job, NULL);
    assert(Demo->JobHandle);
}

int
main(void)
{
    SetProcessDPIAware();

    demo Demo = { 0 };
    InitializeWindow(&Demo);
    Initialize(&Demo);

    for (;;)
    {
        MSG Message = { 0 };
        if (PeekMessage(&Message, 0, 0, 0, PM_REMOVE))
        {
            DispatchMessage(&Message);
            if (Message.message == WM_QUIT)
                break;
        }
        else
        {
            double Time;
            float DeltaTime;
            UpdateFrameTime(Demo.Window, &Time, &DeltaTime);

            s_TileIndex[0] = 0;

            for (uint32_t Index = 0; Index < Demo.CpuCoreCount; ++Index)
                SubmitThreadpoolWork(Demo.JobHandle);

            WaitForThreadpoolWorkCallbacks(Demo.JobHandle, FALSE);

            BitBlt(Demo.WindowDc, 0, 0, k_DemoResolutionX, k_DemoResolutionY, Demo.MemoryDc, 0, 0, SRCCOPY);
        }
    }

    return 0;
}
// vim: ts=4 sw=4 expandtab:
