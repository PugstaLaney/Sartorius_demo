// Cell Segmentation Console - code-behind for MainWindow.xaml.
//
// Two input modes:
//   - Single image (.tif/.png/.jpg) → POST /segment, show overlay + per-cell table
//   - Animated GIF (.gif)           → POST /track_timelapse, cache all frames,
//                                     enable the frame slider so the user can
//                                     scrub through time and watch the data update
//
// The slider scrubs the LOCAL cache, no extra HTTP calls. Backend work happens
// once when the GIF is dropped, scrubbing is instant after that.

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Net.Http;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using Microsoft.Win32;

namespace SegmentationConsole;

// =====================================================================
// View model rows and DTOs
// =====================================================================

/// <summary>
/// One row of the per-cell DataGrid. WPF binds the grid columns to these
/// public property names (CellId, AreaPx, etc.) via the XAML Binding= attribute.
/// </summary>
public class CellRow
{
    public int TrackId { get; set; }       // 0 for single-image mode (no tracking)
    public int CellId { get; set; }
    public int AreaPx { get; set; }
    public double PerimeterPx { get; set; }
    public double Eccentricity { get; set; }
    public double Solidity { get; set; }
    public double CentroidY { get; set; }
    public double CentroidX { get; set; }
}

public class SegmentResponse
{
    [JsonPropertyName("cell_count")] public int CellCount { get; set; }
    [JsonPropertyName("inference_ms")] public double InferenceMs { get; set; }
    [JsonPropertyName("device")] public string Device { get; set; } = "";
    [JsonPropertyName("mask_png_base64")] public string MaskPngBase64 { get; set; } = "";
    [JsonPropertyName("per_cell")] public List<PerCellDto> PerCell { get; set; } = new();
    [JsonPropertyName("summary")] public SummaryDto Summary { get; set; } = new();
}

public class PerCellDto
{
    [JsonPropertyName("cell_id")] public int CellId { get; set; }
    [JsonPropertyName("track_id")] public int TrackId { get; set; }
    [JsonPropertyName("area_px")] public int AreaPx { get; set; }
    [JsonPropertyName("perimeter_px")] public double PerimeterPx { get; set; }
    [JsonPropertyName("eccentricity")] public double Eccentricity { get; set; }
    [JsonPropertyName("solidity")] public double Solidity { get; set; }
    [JsonPropertyName("centroid_y")] public double CentroidY { get; set; }
    [JsonPropertyName("centroid_x")] public double CentroidX { get; set; }
}

public class SummaryDto
{
    [JsonPropertyName("count")] public int Count { get; set; }
    [JsonPropertyName("area_px")] public AreaStats? AreaPx { get; set; }
    [JsonPropertyName("eccentricity")] public EccStats? Eccentricity { get; set; }
}

public class AreaStats
{
    [JsonPropertyName("mean")] public double Mean { get; set; }
    [JsonPropertyName("median")] public double Median { get; set; }
    [JsonPropertyName("min")] public int Min { get; set; }
    [JsonPropertyName("max")] public int Max { get; set; }
}

public class EccStats
{
    [JsonPropertyName("mean")] public double Mean { get; set; }
}

// Shape of the multi-frame response from /track_timelapse.
public class TimelapseResponse
{
    [JsonPropertyName("n_frames")] public int NFrames { get; set; }
    [JsonPropertyName("frames")] public List<TimelapseFrame> Frames { get; set; } = new();
    [JsonPropertyName("tracks_summary")] public TracksSummary TracksSummary { get; set; } = new();
    [JsonPropertyName("device")] public string Device { get; set; } = "";
}

public class TimelapseFrame
{
    [JsonPropertyName("frame_index")] public int FrameIndex { get; set; }
    [JsonPropertyName("inference_ms")] public double InferenceMs { get; set; }
    [JsonPropertyName("cell_count")] public int CellCount { get; set; }
    [JsonPropertyName("original_png_base64")] public string OriginalPngBase64 { get; set; } = "";
    [JsonPropertyName("overlay_png_base64")] public string OverlayPngBase64 { get; set; } = "";
    [JsonPropertyName("per_cell")] public List<PerCellDto> PerCell { get; set; } = new();
    [JsonPropertyName("summary")] public SummaryDto Summary { get; set; } = new();
}

public class TracksSummary
{
    [JsonPropertyName("n_tracks")] public int NTracks { get; set; }
    [JsonPropertyName("full_length")] public int FullLength { get; set; }
    [JsonPropertyName("partial")] public int Partial { get; set; }
}


// =====================================================================
// MainWindow
// =====================================================================

public partial class MainWindow : Window
{
    private const string ApiBase = "http://localhost:8000";
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromMinutes(5) };

    private string? _currentFile;
    private readonly ObservableCollection<CellRow> _cells = new();

    // Cached time-lapse response after processing a GIF. The slider scrubs
    // this in-memory cache; no HTTP calls happen while the user moves the slider.
    private TimelapseResponse? _timelapse;

    public MainWindow()
    {
        InitializeComponent();
        CellGrid.ItemsSource = _cells;
        Loaded += async (_, _) => await CheckBackendHealth();
    }

    // ------------------------------------------------------------------
    // Backend health check on startup. Updates the device badge.
    // ------------------------------------------------------------------
    private async Task CheckBackendHealth()
    {
        try
        {
            var resp = await Http.GetAsync($"{ApiBase}/health");
            if (resp.IsSuccessStatusCode)
            {
                var json = await resp.Content.ReadAsStringAsync();
                using var doc = JsonDocument.Parse(json);
                var device = doc.RootElement.GetProperty("device").GetString() ?? "unknown";
                DeviceBadge.Text = $"device: {device}";
                DeviceBadge.Foreground = (System.Windows.Media.Brush)FindResource("OkBrush");
            }
            else
            {
                SetBackendOffline();
            }
        }
        catch
        {
            SetBackendOffline();
        }
    }

    private void SetBackendOffline()
    {
        DeviceBadge.Text = "device: backend offline";
        DeviceBadge.Foreground = (System.Windows.Media.Brush)FindResource("ErrBrush");
    }

    // ------------------------------------------------------------------
    // Drag-and-drop handlers on the left image panel.
    // ------------------------------------------------------------------
    private static readonly string[] SingleImageExtensions =
        { ".tif", ".tiff", ".png", ".jpg", ".jpeg" };
    private static readonly string[] TimelapseExtensions = { ".gif" };

    private void DropZone_DragEnter(object sender, DragEventArgs e)
    {
        e.Effects = IsAcceptableDrop(e) ? DragDropEffects.Copy : DragDropEffects.None;
        e.Handled = true;
    }

    private void DropZone_DragOver(object sender, DragEventArgs e)
    {
        e.Effects = IsAcceptableDrop(e) ? DragDropEffects.Copy : DragDropEffects.None;
        e.Handled = true;
    }

    private void DropZone_DragLeave(object sender, DragEventArgs e)
    {
        // No-op. Hook for a future "hover" style if we want one.
    }

    private static bool IsAcceptableDrop(DragEventArgs e)
    {
        if (!e.Data.GetDataPresent(DataFormats.FileDrop)) return false;
        if (e.Data.GetData(DataFormats.FileDrop) is not string[] files || files.Length == 0)
            return false;
        var ext = Path.GetExtension(files[0]).ToLowerInvariant();
        return Array.IndexOf(SingleImageExtensions, ext) >= 0
            || Array.IndexOf(TimelapseExtensions, ext) >= 0;
    }

    private void DropZone_Drop(object sender, DragEventArgs e)
    {
        if (e.Data.GetData(DataFormats.FileDrop) is string[] files && files.Length > 0)
        {
            LoadFile(files[0]);
        }
    }

    private void DropZone_Click(object sender, MouseButtonEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Filter = "Images and GIFs|*.tif;*.tiff;*.png;*.jpg;*.jpeg;*.gif|All files|*.*",
        };
        if (dlg.ShowDialog(this) == true)
        {
            LoadFile(dlg.FileName);
        }
    }

    // ------------------------------------------------------------------
    // Loading a file into the UI. The Run button label adapts to the mode.
    // ------------------------------------------------------------------
    private void LoadFile(string path)
    {
        try
        {
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.UriSource = new Uri(path);
            bitmap.EndInit();
            bitmap.Freeze();

            OriginalImage.Source = bitmap;
            DropZoneHint.Visibility = Visibility.Collapsed;
            OverlayImage.Source = null;
            OverlayHint.Visibility = Visibility.Visible;

            _currentFile = path;
            _timelapse = null;
            FilePathLabel.Text = path;
            RunButton.IsEnabled = true;

            // Hide the slider until a new time-lapse is processed.
            TimelapsePanel.Visibility = Visibility.Collapsed;
            _cells.Clear();

            RunButton.Content = IsGif(path) ? "Run tracking" : "Run segmentation";
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, $"Could not load image:\n{ex.Message}",
                "Load error", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private static bool IsGif(string path) =>
        string.Equals(Path.GetExtension(path), ".gif", StringComparison.OrdinalIgnoreCase);

    // ------------------------------------------------------------------
    // Run button. Branches on input type.
    // ------------------------------------------------------------------
    private async void RunButton_Click(object sender, RoutedEventArgs e)
    {
        if (_currentFile == null) return;

        RunButton.IsEnabled = false;
        var originalLabel = RunButton.Content;
        RunButton.Content = "Running...";

        try
        {
            if (IsGif(_currentFile))
            {
                await RunTimelapse(_currentFile);
            }
            else
            {
                await RunSingleImage(_currentFile);
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, $"Request failed:\n{ex.Message}",
                "Network error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            RunButton.IsEnabled = true;
            RunButton.Content = originalLabel;
        }
    }

    private async Task RunSingleImage(string path)
    {
        using var form = new MultipartFormDataContent();
        await using var stream = File.OpenRead(path);
        form.Add(new StreamContent(stream), "file", Path.GetFileName(path));

        var resp = await Http.PostAsync($"{ApiBase}/segment", form);
        var body = await resp.Content.ReadAsStringAsync();

        if (!resp.IsSuccessStatusCode)
        {
            MessageBox.Show(this, $"Backend returned {(int)resp.StatusCode}:\n{body}",
                "Inference failed", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        var result = JsonSerializer.Deserialize<SegmentResponse>(body)
                     ?? throw new InvalidOperationException("Empty response");

        ApplySingleResult(result);
    }

    private async Task RunTimelapse(string path)
    {
        using var form = new MultipartFormDataContent();
        await using var stream = File.OpenRead(path);
        form.Add(new StreamContent(stream), "file", Path.GetFileName(path));

        var resp = await Http.PostAsync($"{ApiBase}/track_timelapse", form);
        var body = await resp.Content.ReadAsStringAsync();

        if (!resp.IsSuccessStatusCode)
        {
            MessageBox.Show(this, $"Backend returned {(int)resp.StatusCode}:\n{body}",
                "Tracking failed", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        var result = JsonSerializer.Deserialize<TimelapseResponse>(body)
                     ?? throw new InvalidOperationException("Empty response");

        _timelapse = result;

        // Wire up the slider for the new frame count.
        TimelapsePanel.Visibility = Visibility.Visible;
        FrameSlider.Maximum = Math.Max(0, result.NFrames - 1);
        FrameSlider.Value = 0;
        DeviceBadge.Text = $"device: {result.Device}";
        DeviceBadge.Foreground = (System.Windows.Media.Brush)FindResource("OkBrush");

        // Show frame 0 immediately. Slider's ValueChanged handles the rest.
        RenderFrame(0);
    }

    // ------------------------------------------------------------------
    // Single-image path: just show one segmentation result.
    // ------------------------------------------------------------------
    private void ApplySingleResult(SegmentResponse r)
    {
        var maskImg = DecodeBase64Png(r.MaskPngBase64);
        OverlayImage.Source = maskImg;
        OverlayHint.Visibility = Visibility.Collapsed;

        CountBadge.Text = $"cells: {r.CellCount}";
        LatencyBadge.Text = $"latency: {r.InferenceMs:F0} ms";
        DeviceBadge.Text = $"device: {r.Device}";
        DeviceBadge.Foreground = (System.Windows.Media.Brush)FindResource("OkBrush");

        SetSummaryText(r.Summary, r.CellCount);
        FillCellGrid(r.PerCell);
    }

    // ------------------------------------------------------------------
    // Slider ValueChanged → re-render the UI from the cached frame.
    // ------------------------------------------------------------------
    private void FrameSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (_timelapse == null) return;
        RenderFrame((int)Math.Round(e.NewValue));
    }

    private void RenderFrame(int frameIndex)
    {
        if (_timelapse == null) return;
        if (frameIndex < 0 || frameIndex >= _timelapse.Frames.Count) return;

        var f = _timelapse.Frames[frameIndex];

        OriginalImage.Source = DecodeBase64Png(f.OriginalPngBase64);
        OverlayImage.Source = DecodeBase64Png(f.OverlayPngBase64);
        OverlayHint.Visibility = Visibility.Collapsed;

        CountBadge.Text = $"cells: {f.CellCount}";
        LatencyBadge.Text = $"latency: {f.InferenceMs:F0} ms";

        FrameCounter.Text =
            $"{frameIndex + 1} / {_timelapse.NFrames}    " +
            $"tracks: {_timelapse.TracksSummary.NTracks} " +
            $"({_timelapse.TracksSummary.FullLength} full-length)";

        SetSummaryText(f.Summary, f.CellCount);
        FillCellGrid(f.PerCell);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    private static BitmapImage DecodeBase64Png(string base64)
    {
        var bytes = Convert.FromBase64String(base64);
        var img = new BitmapImage();
        img.BeginInit();
        img.CacheOption = BitmapCacheOption.OnLoad;
        img.StreamSource = new MemoryStream(bytes);
        img.EndInit();
        img.Freeze();
        return img;
    }

    private void SetSummaryText(SummaryDto? s, int fallbackCount)
    {
        if (s?.AreaPx != null && s?.Eccentricity != null)
        {
            SummaryText.Text =
                $"count={s.Count}    " +
                $"area_px: mean={s.AreaPx.Mean:F0}  median={s.AreaPx.Median:F0}  " +
                $"min={s.AreaPx.Min}  max={s.AreaPx.Max}    " +
                $"eccentricity_mean={s.Eccentricity.Mean:F3}";
        }
        else
        {
            SummaryText.Text = $"count={fallbackCount}";
        }
    }

    private void FillCellGrid(List<PerCellDto> rows)
    {
        _cells.Clear();
        foreach (var c in rows)
        {
            _cells.Add(new CellRow
            {
                TrackId = c.TrackId,
                CellId = c.CellId,
                AreaPx = c.AreaPx,
                PerimeterPx = c.PerimeterPx,
                Eccentricity = c.Eccentricity,
                Solidity = c.Solidity,
                CentroidY = c.CentroidY,
                CentroidX = c.CentroidX,
            });
        }
    }
}
