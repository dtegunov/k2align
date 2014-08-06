using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Globalization;

namespace K2AlignDaemon
{
    public partial class MainWindow : MahApps.Metro.Controls.MetroWindow
    {
        Task ProcessingTask = null;
        Task AsyncWriteTask = null;
        Task AsyncArchiveTask = null;

        LogWriter Log = null;

        ProcessingModes Mode = ProcessingModes.Daemon;
        bool IsRunning = false;
        ProcessingOptions Options = new ProcessingOptions();

        public MainWindow()
        {
            InitializeComponent();
            Closing += MainWindow_Closing;
            Options.PropertyChanged += Options_PropertyChanged;

            if (!Directory.Exists("C:\\K2\\temp"))
                Directory.CreateDirectory("C:\\K2\\temp");

            List<string> TempFiles = new List<string>();
            foreach (var Path in Directory.EnumerateFiles("C:\\K2\\temp"))
                TempFiles.Add(Path);
            if (TempFiles.Count > 0)
            { 
                if(MessageBox.Show("There are some files in C:\\K2\\temp. Delete them?", "Maintenance", MessageBoxButton.YesNo) == MessageBoxResult.Yes)
                    foreach (var Path in TempFiles)
                    {
                        try
                        {
                            File.Delete(Path);
                        }
                        catch
                        {
                            MessageBox.Show("Could not delete " + Path);
                        }
                    }
            }


            DataContext = Options;
            Options.OutputPath = Properties.Settings.Default.OutputPath;
            Options.ArchivePath = Properties.Settings.Default.ArchivePath;
            Options.InFormatRaw = Properties.Settings.Default.InFormatRaw;
            Options.InFormatMrc = Properties.Settings.Default.InFormatMrc;
            Options.InFormatEm = Properties.Settings.Default.InFormatEm;
            Options.OutFormatMrc = Properties.Settings.Default.OutFormatMrc;
            Options.OutFormatEm = Properties.Settings.Default.OutFormatEm;
            Options.OutFormatMRC16bit = Properties.Settings.Default.OutFormatMRC16bit;
            Options.OutFormatMRC32bit = Properties.Settings.Default.OutFormatMRC32bit;
            Options.RawWidth = Properties.Settings.Default.RawWidth;
            Options.RawHeight = Properties.Settings.Default.RawHeight;
            Options.RawDepth = Properties.Settings.Default.RawDepth;
            Options.ArchiveZip = Properties.Settings.Default.ArchiveZip;
            Options.ArchiveKeep = Properties.Settings.Default.ArchiveKeep;
            Options.CorrectGain = Properties.Settings.Default.CorrectGain;
            Options.GainPath = Properties.Settings.Default.GainPath;
            Options.CorrectXray = Properties.Settings.Default.CorrectXray;
            Options.BandpassLow = Properties.Settings.Default.BandpassLow;
            Options.BandpassHigh = Properties.Settings.Default.BandpassHigh;
            Options.ProcessFirst = Properties.Settings.Default.ProcessFirst;
            Options.ProcessLast = Properties.Settings.Default.ProcessLast;
            Options.OutputFirst = Properties.Settings.Default.OutputFirst;
            Options.OutputRanges = Properties.Settings.Default.OutputRanges;
            Options.OutputLast = Properties.Settings.Default.OutputLast;
            Options.MaxDrift = Properties.Settings.Default.MaxDrift;
            Options.MinValidFrames = Properties.Settings.Default.MinValidFrames;
            Options.AverageWindow = Properties.Settings.Default.AverageWindow;
            Options.QuadsX = Properties.Settings.Default.QuadsX;
            Options.QuadsY = Properties.Settings.Default.QuadsY;
            Options.QuadSize = Properties.Settings.Default.QuadSize;
            Options.OutputDownsample = Properties.Settings.Default.OutputDownsample;
            Options.DeletePatterns = Properties.Settings.Default.DeletePatterns;
            Options.DeleteFolders = Properties.Settings.Default.DeleteFolders;

            Log = new LogWriter("C:\\K2\\log.txt");
            Log.Write(string.Format("Session started by {0}", System.Security.Principal.WindowsIdentity.GetCurrent().Name));
        }

        void Options_PropertyChanged(object sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (e.PropertyName == "OutputRanges")
            {
                if (Options.NumberOutputRanges <= 1)
                {
                    PanelOutputRangeSliders.Visibility = Visibility.Visible;
                    PanelOutputRangeText.Visibility = Visibility.Collapsed;
                }
                else
                {
                    PanelOutputRangeSliders.Visibility = Visibility.Collapsed;
                    PanelOutputRangeText.Visibility = Visibility.Visible;
                }
            }
        }

        void MainWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            Properties.Settings.Default.OutputPath = Options.OutputPath;
            Properties.Settings.Default.ArchivePath = Options.ArchivePath;
            Properties.Settings.Default.InFormatRaw = Options.InFormatRaw;
            Properties.Settings.Default.InFormatMrc = Options.InFormatMrc;
            Properties.Settings.Default.InFormatEm = Options.InFormatEm;
            Properties.Settings.Default.OutFormatMrc = Options.OutFormatMrc;
            Properties.Settings.Default.OutFormatEm = Options.OutFormatEm;
            Properties.Settings.Default.OutFormatMRC16bit = Options.OutFormatMRC16bit;
            Properties.Settings.Default.OutFormatMRC32bit = Options.OutFormatMRC32bit;
            Properties.Settings.Default.RawWidth = Options.RawWidth;
            Properties.Settings.Default.RawHeight = Options.RawHeight;
            Properties.Settings.Default.RawDepth = Options.RawDepth;
            Properties.Settings.Default.ArchiveZip = Options.ArchiveZip;
            Properties.Settings.Default.ArchiveKeep = Options.ArchiveKeep;
            Properties.Settings.Default.CorrectGain = Options.CorrectGain;
            Properties.Settings.Default.GainPath = Options.GainPath;
            Properties.Settings.Default.CorrectXray = Options.CorrectXray;
            Properties.Settings.Default.BandpassLow = Options.BandpassLow;
            Properties.Settings.Default.BandpassHigh = Options.BandpassHigh;
            Properties.Settings.Default.ProcessFirst = Options.ProcessFirst;
            Properties.Settings.Default.ProcessLast = Options.ProcessLast;
            Properties.Settings.Default.OutputFirst = Options.OutputFirst;
            Properties.Settings.Default.OutputLast = Options.OutputLast;
            Properties.Settings.Default.OutputRanges = Options.OutputRanges;
            Properties.Settings.Default.MaxDrift = Options.MaxDrift;
            Properties.Settings.Default.MinValidFrames = Options.MinValidFrames;
            Properties.Settings.Default.AverageWindow = Options.AverageWindow;
            Properties.Settings.Default.QuadsX = Options.QuadsX;
            Properties.Settings.Default.QuadsY = Options.QuadsY;
            Properties.Settings.Default.QuadSize = Options.QuadSize;
            Properties.Settings.Default.OutputDownsample = Options.OutputDownsample;
            Properties.Settings.Default.DeletePatterns = Options.DeletePatterns;
            Properties.Settings.Default.DeleteFolders = Options.DeleteFolders;

            Properties.Settings.Default.Save();
            Log.Dispose();
        }

        /// <summary>
        /// Folder processing has been selected in the start screen
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void ButtonModeFolder_Checked(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog Dialog = new System.Windows.Forms.FolderBrowserDialog();
            Dialog.SelectedPath = Options.InputPath;
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            while (true)
                if (Result.ToString() == "OK")
                {
                    TabOptions.Visibility = Visibility.Visible;
                    TabsMain.SelectedIndex = 1;
                    if (Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '\\' && Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '/')
                        Dialog.SelectedPath += '/';

                    try
                    {
                        foreach (var Path in Directory.EnumerateFiles(Dialog.SelectedPath, "*.*", SearchOption.TopDirectoryOnly))
                        {
                            Stream S = File.OpenRead(Path);
                            S.Close();
                        }
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show("Couldn't read one of the files in folder, error message: " + ex.Message);
                        Log.Write(ex);
                        continue;
                    }

                    AdjustToFolder(Dialog.SelectedPath);

                    Mode = ProcessingModes.Folder;
                    ProgressIndicator.IsIndeterminate = false;
                    PanelDelete.Visibility = Visibility.Collapsed;
                    this.Height = 730;

                    Log.Write("Doing folder mode in " + Dialog.SelectedPath);
                    break;
                }
                else
                    break;
        }

        /// <summary>
        /// On-the-fly processing has been selected in the start screen
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void ButtonModeDaemon_Checked(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog Dialog = new System.Windows.Forms.FolderBrowserDialog();
            Dialog.SelectedPath = Options.InputPath;
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            while (true)
                if (Result.ToString() == "OK")
                {
                    TabOptions.Visibility = Visibility.Visible;
                    TabsMain.SelectedIndex = 1;
                    if (Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '\\' && Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '/')
                        Dialog.SelectedPath += '/';

                    try
                    {
                        foreach (var Path in Directory.EnumerateFiles(Dialog.SelectedPath, "*.*", SearchOption.TopDirectoryOnly))
                        {
                            Stream S = File.OpenRead(Path);
                            S.Close();
                        }
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show("Couldn't read one of the files in folder, error message: " + ex.Message);
                        Log.Write(ex);
                        continue;
                    }

                    AdjustToFolder(Dialog.SelectedPath);

                    TabOptions.Visibility = Visibility.Visible;
                    TabsMain.SelectedIndex = 1;
                    Mode = ProcessingModes.Daemon;
                    ProgressIndicator.IsIndeterminate = true;
                    PanelDelete.Visibility = Visibility.Visible;
                    this.Height = 800;

                    Log.Write("Doing on-the-fly mode in " + Dialog.SelectedPath);
                    break;
                }
                else
                    break;
        }

        /// <summary>
        /// Attempts to adjust processing settings to the files present in the selected folder.
        /// </summary>
        /// <param name="path">Selected folder path</param>
        private void AdjustToFolder(string path)
        {
            Options.InputPath = path;

            int3 FrameDims = new int3(1, 1, 1);
            if (Directory.EnumerateFiles(Options.InputPath, "*.mrc").Count() > 0)
            {
                Options.InFormatMrc = true;
                Options.InFormatEm = false;
                Options.InFormatRaw = false;

                Options.OutFormatMrc = true;
                Options.OutFormatEm = false;

                IEnumerable<string> Files = Directory.EnumerateFiles(Options.InputPath, "*.mrc");
                foreach (string Filename in Files)
                {
                    FileInfo Info = new FileInfo(Filename);
                    using (BinaryReader Reader = new BinaryReader(File.OpenRead(Info.FullName)))
                    {
                        byte[] Buffer = Reader.ReadBytes(512);
                        unsafe
                        {
                            fixed (byte* BufferPtr = Buffer)
                            {
                                FrameDims.X = ((int*)BufferPtr)[0];
                                FrameDims.Y = ((int*)BufferPtr)[1];
                                FrameDims.Z = ((int*)BufferPtr)[2];
                            }
                        }
                    }
                    break;
                }
            }
            else if (Directory.EnumerateFiles(Options.InputPath, "*.em").Count() > 0)
            {
                Options.InFormatMrc = false;
                Options.InFormatEm = true;
                Options.InFormatRaw = false;

                Options.OutFormatMrc = false;
                Options.OutFormatEm = true;

                IEnumerable<string> Files = Directory.EnumerateFiles(Options.InputPath, "*.em");
                foreach (string Filename in Files)
                {
                    FileInfo Info = new FileInfo(Filename);
                    using (BinaryReader Reader = new BinaryReader(File.OpenRead(Info.FullName)))
                    {
                        byte[] Buffer = Reader.ReadBytes(512);
                        unsafe
                        {
                            fixed (byte* BufferPtr = Buffer)
                            {
                                FrameDims.X = ((int*)BufferPtr)[1];
                                FrameDims.Y = ((int*)BufferPtr)[2];
                                FrameDims.Z = ((int*)BufferPtr)[3];
                            }
                        }
                    }
                    break;
                }
            }
            else if (Directory.EnumerateFiles(Options.InputPath, "*.dat").Count() > 0)
            {
                Options.InFormatMrc = false;
                Options.InFormatEm = false;
                Options.InFormatRaw = true;

                FrameDims = new int3(Options.RawWidth, Options.RawHeight, Options.RawDepth);
            }
            else if (Directory.EnumerateFiles(Options.InputPath, "*.raw").Count() > 0)
            {
                Options.InFormatMrc = false;
                Options.InFormatEm = false;
                Options.InFormatRaw = true;

                FrameDims = new int3(4096, 4096, 7);
                Options.RawWidth = 4096;
                Options.RawHeight = 4096;
                Options.RawDepth = 14;
                Options.CorrectGain = false;
                Options.OutputFirst = 1;
                Options.OutputLast = 7;
                Options.ProcessFirst = 1;
                Options.ProcessLast = 14;
            }

            if (FrameDims.X == 7676 && FrameDims.Y == 7420)
                Options.GainPath = "C:\\K2\\gain_7676x7420.em";
            else if (FrameDims.Y == 7676 && FrameDims.X == 7420)
                Options.GainPath = "C:\\K2\\gain_7420x7676.em";
            if (FrameDims.X == 3838 && FrameDims.Y == 3710)
                Options.GainPath = "C:\\K2\\gain_3838x3710.em";
            else if (FrameDims.Y == 3838 && FrameDims.X == 3710)
                Options.GainPath = "C:\\K2\\gain_3710x3838.em";
        }

        private void CheckGain_Checked(object sender, RoutedEventArgs e)
        {
            GridGainPath.Visibility = Visibility.Visible;
        }

        private void CheckGain_Unchecked(object sender, RoutedEventArgs e)
        {
            GridGainPath.Visibility = Visibility.Collapsed;
        }

        private void RadioArchiveZip_Checked(object sender, RoutedEventArgs e)
        {
            GridTransferPath.Visibility = Visibility.Visible;
        }

        private void RadioArchiveZip_Unchecked(object sender, RoutedEventArgs e)
        {
            GridTransferPath.Visibility = Visibility.Collapsed;
        }

        private void RadioInFormatRaw_Checked(object sender, RoutedEventArgs e)
        {
            StackRawDims.Visibility = Visibility.Visible;
        }

        private void RadioInFormatRaw_Unchecked(object sender, RoutedEventArgs e)
        {
            StackRawDims.Visibility = Visibility.Collapsed;
        }

        private void ButtonOutputpath_Click(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog Dialog = new System.Windows.Forms.FolderBrowserDialog();
            Dialog.SelectedPath = Options.OutputPath;
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            while (true)
                if (Result.ToString() == "OK")
                {
                    if (Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '\\')
                        Dialog.SelectedPath += '\\';

                    try
                    {
                        Stream S = File.Create(Dialog.SelectedPath + "test.test");
                        S.Close();
                        File.Delete(Dialog.SelectedPath + "test.test");
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show("Couldn't write to the selected folder, error message: " + ex.Message);
                        Log.Write(ex);
                        continue;
                    }

                    Options.OutputPath = Dialog.SelectedPath;
                    break;
                }
                else
                    break;
        }

        private void ButtonArchivepath_Click(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.FolderBrowserDialog Dialog = new System.Windows.Forms.FolderBrowserDialog();
            Dialog.SelectedPath = Options.ArchivePath;
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();
            
            while (true)
                if (Result.ToString() == "OK")
                {
                    if (Dialog.SelectedPath[Dialog.SelectedPath.Length - 1] != '\\')
                        Dialog.SelectedPath += '\\';

                    try
                    {
                        Stream S = File.Create(Dialog.SelectedPath + "test.test");
                        S.Close();
                        File.Delete(Dialog.SelectedPath + "test.test");
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show("Couldn't write to the selected folder, error message: " + ex.Message);
                        Log.Write(ex);
                        continue;
                    }

                    Options.ArchivePath = Dialog.SelectedPath;
                    break;
                }
                else
                    break;
        }

        private void ButtonGainpath_Click(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog();
            Dialog.Filter = "EM Files|*.em";
            Dialog.Multiselect = false;
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                Options.GainPath = Dialog.FileName;
                Options.CustomGain = true;
            }
        }

        private void ButtonProcess_Click(object sender, RoutedEventArgs e)
        {
            if (IsRunning)
            {
                ButtonProcess.Content = "STOPPING...";
                ButtonProcess.IsEnabled = false;
                IsRunning = false;
                return;
            }

            //IsRunning = true;
            ProcessingTask = new Task(() =>
                {
                    List<string> ProcessedFiles = new List<string>();

                    string FileExtension = "";
                    string FileWildcard = "*";
                    if (Options.InFormatRaw)
                        FileExtension = "dat";
                    else if (Options.InFormatMrc)
                        FileExtension = "mrc";
                    else if (Options.InFormatEm)
                        FileExtension = "em";

                    string OutputFileExtension = "";
                    if (Options.OutFormatMrc)
                        OutputFileExtension = "mrc";
                    else if (Options.OutFormatEm)
                        OutputFileExtension = "em";

                    if (Options.InputPath[Options.InputPath.Length - 1] != '\\' && Options.InputPath[Options.InputPath.Length - 1] != '/')
                        Options.InputPath += '\\';
                    if(Options.ArchiveZip)
                        if (Options.ArchivePath[Options.ArchivePath.Length - 1] != '\\' && Options.ArchivePath[Options.ArchivePath.Length - 1] != '/')
                            Options.ArchivePath += '\\';
                    
                    float[] GainMask = new float[1];
                    int3 GainDims = new int3(1, 1, 1);
                    if(Options.CorrectGain)
                    {
                        using(BinaryReader Reader = new BinaryReader(File.OpenRead(Options.GainPath)))
                        {
                            byte[] Buffer = Reader.ReadBytes(512);
                            unsafe
                            {
                                fixed (byte* BufferPtr = Buffer)
                                { 
                                    GainDims.X = ((int*)BufferPtr)[1];
                                    GainDims.Y = ((int*)BufferPtr)[2];
                                    GainDims.Z = ((int*)BufferPtr)[3];
                                }
                            }

                            GainMask = new float[(uint)GainDims.Elements()];
                            Buffer = Reader.ReadBytes((int)GainDims.Elements() * sizeof(float));
                            unsafe
                            {
                                fixed (byte* BufferPtr = Buffer)
                                fixed (float* GainPtr = GainMask)
                                {
                                    float* BufferP = (float*)BufferPtr;
                                    float* GainP = GainPtr;
                                    for (int i = 0; i < GainMask.Length; i++) 
			                            *GainP++ = *BufferP++;
                                }
                            }
                        }
                    }

                    using(TextWriter CorruptWriter = new StreamWriter(File.Create(Options.OutputPath + "corrupt.txt")))
                        while (IsRunning)
                        {

                            if (Options.InFormatRaw && Directory.EnumerateFiles(Options.InputPath, "*_n0.raw").Count() > 0)
                            {
                                FileExtension = "raw";
                                FileWildcard = "*_n0";
                            }

                            IEnumerable<string> VolatileFiles = Directory.EnumerateFiles(Options.InputPath, FileWildcard + "." + FileExtension);
                            Dictionary<string, long> FileSizes = new Dictionary<string, long>();
                            List<string> Files = new List<string>();

                            if (FileExtension == "raw")
                            {
                                while (true)
                                {
                                    if (!IsRunning)
                                        break;

                                    bool UnprocessedFound = false;
                                    foreach (string Filename in VolatileFiles)
                                    {
                                        if (!ProcessedFiles.Contains(Filename))
                                        {
                                            UnprocessedFound = true;
                                            break;
                                        }
                                    }
                                    if (!UnprocessedFound)
                                        break;

                                    int CombinedFilesNeeded = (Options.ProcessLast - Options.ProcessFirst + 1 + 6) / 7;
                                    List<string> CombinedNames = new List<string>();
                                    DateTime GroupStartDate = new DateTime(1950, 1, 1);

                                    while (CombinedNames.Count < CombinedFilesNeeded)
                                    {
                                        if (!IsRunning)
                                            break;

                                        foreach (string Filename in VolatileFiles)
                                        {
                                            if (ProcessedFiles.Contains(Filename))
                                                continue;
                                            ProcessedFiles.Add(Filename);

                                            FileInfo Info = new FileInfo(Filename);
                                            string DatePart = Info.Name.Substring(Info.Name.IndexOf('_') + 1, 15);
                                            DateTime FileDate = DateTime.ParseExact(DatePart, "yyyyMMdd_HHmmss", CultureInfo.InvariantCulture.DateTimeFormat);
                                            TimeSpan Difference = FileDate - GroupStartDate;
                                            if (Difference.TotalSeconds > 28.0 * (double)CombinedNames.Count)
                                            {
                                                CombinedNames.Clear();
                                                CombinedNames.Add(Filename);
                                                GroupStartDate = FileDate;
                                            }
                                            else
                                                CombinedNames.Add(Filename);

                                            break;
                                        }
                                        if (CombinedNames.Count < CombinedFilesNeeded)
                                            Thread.Sleep(50);
                                    }
                                    if (!IsRunning)
                                        break;

                                    FileInfo RootInfo = new FileInfo(CombinedNames[0]);

                                    string NameRoot = RootInfo.Name.Substring(0, RootInfo.Name.IndexOf("_n0.raw"));

                                    byte[] Header = new byte[512];
                                    using (BinaryReader Reader = new BinaryReader(File.OpenRead("C:\\K2\\emdummy.em")))
                                        Header = Reader.ReadBytes(Header.Length);

                                    using (BinaryWriter Writer = new BinaryWriter(File.Create("C:\\K2\\temp\\" + NameRoot + ".em")))
                                    {
                                        byte[] LocalHeader = new byte[Header.Length];
                                        Array.Copy(Header, LocalHeader, Header.Length);

                                        unsafe
                                        {
                                            fixed (byte* LocalHeaderPtr = LocalHeader)
                                            {
                                                ((int*)LocalHeaderPtr)[1] = Options.RawWidth;
                                                ((int*)LocalHeaderPtr)[2] = Options.RawHeight;
                                                ((int*)LocalHeaderPtr)[3] = CombinedFilesNeeded * 7;
                                                LocalHeaderPtr[3] = 4;
                                            }
                                        }
                                        Writer.Write(LocalHeader);

                                        foreach (string Filename in CombinedNames)
                                        {
                                            FileInfo Info = new FileInfo(Filename);
                                            string LocalNameRoot = Info.Name.Substring(0, RootInfo.Name.IndexOf("_n0.raw"));
                                            string LastFrameName = LocalNameRoot + "_n6.raw";
                                            while (Directory.EnumerateFiles(Options.InputPath, LastFrameName).Count() == 0)
                                                Thread.Sleep(50);
                                            Thread.Sleep(300);

                                            for (int i = 0; i < 7; i++)
                                            {
                                                string FrameName = LocalNameRoot + "_n" + i + ".raw";
                                                while (Directory.EnumerateFiles(Options.InputPath, FrameName).Count() == 0)
                                                    Thread.Sleep(100);
                                                bool Success = false;
                                                int FailCount = 0;
                                                while (!Success)
                                                {
                                                    try
                                                    {
                                                        File.Copy(Options.InputPath + FrameName, "C:\\K2\\temp\\" + FrameName);
                                                        using (BinaryReader Reader = new BinaryReader(File.OpenRead("C:\\K2\\temp\\" + FrameName)))
                                                        {
                                                            Reader.ReadBytes(49);
                                                            byte[] Buffer = Reader.ReadBytes(Options.RawWidth * Options.RawHeight * sizeof(float));
                                                            Writer.Write(Buffer);
                                                        }
                                                        Success = true;
                                                    }
                                                    catch
                                                    {
                                                        Thread.Sleep(2000);
                                                        FailCount++;
                                                        if (FailCount % 50 == 0)
                                                            MessageBox.Show("Failed to read frame after 50 attempts. Please fix the problem and press OK to continue.");
                                                    }
                                                }
                                                //LogWriter.WriteLine("Deleting " + Options.InputPath + FrameName);
                                                //LogWriter.Flush();
                                                Success = false;
                                                FailCount = 0;
                                                while (!Success)
                                                {
                                                    try
                                                    {
                                                        File.Delete("C:\\K2\\temp\\" + FrameName);
                                                        File.Delete(Options.InputPath + FrameName);
                                                        Success = true;
                                                    }
                                                    catch
                                                    {
                                                        Thread.Sleep(2000);
                                                        FailCount++;
                                                        if (FailCount % 50 == 0)
                                                            MessageBox.Show("Failed to delete source frame after 50 attempts. Please fix the problem and press OK to continue.");
                                                    }
                                                }
                                            }
                                        }
                                        Log.Write("Created C:\\K2\\temp\\" + NameRoot + ".em");
                                    }

                                    Files.Add("C:\\K2\\temp\\" + NameRoot + ".em");
                                    if (Mode == ProcessingModes.Daemon)
                                        break;
                                }
                            }
                            else
                            {
                                foreach (string Filename in VolatileFiles)
                                {
                                    FileInfo Info = new FileInfo(Filename);
                                    FileSizes.Add(Filename, Info.Length);
                                }
                            }

                            if (Mode == ProcessingModes.Daemon)
                            {
                                Thread.Sleep(1000);
                            }
                            foreach (string Filename in VolatileFiles)
                            {
                                FileInfo Info = new FileInfo(Filename);
                                if (FileSizes.ContainsKey(Filename) && Info.Length == FileSizes[Filename])
                                    Files.Add(Filename);
                            }

                            ProgressIndicator.Dispatcher.Invoke(() => 
                                {
                                    ProgressIndicator.Visibility = Visibility.Visible;
                                    ProgressIndicator.Value = 0;
                                    ProgressIndicator.Maximum = Files.Count();
                                });
                            foreach (string Filename in Files)
                            {
                                try
                                {
                                    //LogWriter.WriteLine("Processing " + Filename);
                                    //LogWriter.Flush();
                                    FileInfo Info = new FileInfo(Filename);
                                    bool SavedTemporary = false;

                                    if (Info.FullName[0].ToString().ToLower() != "c")
                                    {
                                        File.Copy(Info.FullName, "C:\\K2\\temp\\" + Info.Name, true);
                                        SavedTemporary = true;
                                        Info = new FileInfo("C:\\K2\\temp\\" + Info.Name);
                                    }

                                    int RawDataType = 1;
                                    if (ProcessedFiles.Contains(Info.FullName))
                                        continue;
                                    int3 FrameDims = new int3(1, 1, 1);

                                    using (BinaryReader Reader = new BinaryReader(File.OpenRead(Info.FullName)))
                                    {
                                        byte[] Buffer = Reader.ReadBytes(512);
                                        unsafe
                                        {
                                            fixed (byte* BufferPtr = Buffer)
                                            {
                                                if (Options.InFormatMrc)
                                                {
                                                    FrameDims.X = ((int*)BufferPtr)[0];
                                                    FrameDims.Y = ((int*)BufferPtr)[1];
                                                    FrameDims.Z = ((int*)BufferPtr)[2];
                                                }
                                                else if (Options.InFormatEm || FileExtension == "raw")
                                                {
                                                    FrameDims.X = ((int*)BufferPtr)[1];
                                                    FrameDims.Y = ((int*)BufferPtr)[2];
                                                    FrameDims.Z = ((int*)BufferPtr)[3];
                                                }
                                                else if (Options.InFormatRaw)
                                                {
                                                    FrameDims = new int3(Options.RawWidth, Options.RawHeight, Options.RawDepth);
                                                }
                                            }
                                        }
                                    }

                                    if (FileExtension == "dat")
                                    {
                                        ulong BytesIfChar = FrameDims.Elements();
                                        if (Info.Length == (long)BytesIfChar * 4)
                                            RawDataType = 4;
                                        else if (Info.Length < (long)BytesIfChar)
                                        {
                                            ProcessedFiles.Add(Info.FullName);
                                            if(SavedTemporary)
                                                File.Delete(Info.FullName);
                                            continue;
                                        }
                                    }

                                    int3 SubframeDims = new int3(FrameDims.X, FrameDims.Y, 1);
                                    bool[] IsCorrupt = new bool[] { false };
                                    float[] OutputAverage = new float[(uint)SubframeDims.Elements() / (Options.OutputDownsample * Options.OutputDownsample) * Options.NumberOutputRanges];
                                    float[] OutputQuadAverages = new float[Math.Max(1, Options.QuadSize * Options.QuadSize * Options.QuadsX * Options.QuadsY / (Options.OutputDownsample * Options.OutputDownsample)) * Options.NumberOutputRanges];

                                    GPU.FrameAlign(Info.FullName,
                                                   OutputAverage,
                                                   OutputQuadAverages,
                                                   Options.CorrectGain,
                                                   GainMask,
                                                   GainDims,
                                                   Options.CorrectXray,
                                                   (float)Options.BandpassLow,
                                                   (float)Options.BandpassHigh,
                                                   FileExtension == "raw" ? "em" : FileExtension,
                                                   RawDataType,
                                                   Options.InFormatRaw ? FrameDims : new int3(1, 1, 1),
                                                   Options.AverageWindow / 2,
                                                   Math.Max(Options.AverageWindow / 2, 1),
                                                   Options.ProcessFirst - 1,
                                                   Options.ProcessLast - 1,
                                                   Options.OutputRangesIntArray,
                                                   Options.NumberOutputRanges,
                                                   (float)Options.MaxDrift,
                                                   Options.MinValidFrames,
                                                   Options.OutputDownsample,
                                                   new int3(Options.QuadSize, Options.QuadSize, 1),
                                                   new int3(Options.QuadsX, Options.QuadsY, 1),
                                                   IsCorrupt,
                                                   Options.OutputPath + Info.Name + ".log");
                                    //LogWriter.WriteLine("Processed " + Filename);
                                    //LogWriter.Flush();

                                    if (IsCorrupt[0])
                                        CorruptWriter.WriteLine(Info.Name);

                                    byte[] Header = new byte[1];
                                    string HeaderPath = "";
                                    if (FileExtension == OutputFileExtension)
                                        HeaderPath = Info.FullName;
                                    else if (Options.OutFormatEm)
                                        HeaderPath = "C:\\K2\\emdummy.em";
                                    else if (Options.OutFormatMrc)
                                        HeaderPath = "C:\\K2\\mrcdummy.mrc";

                                    using (BinaryReader Reader = new BinaryReader(File.OpenRead(HeaderPath)))
                                    {
                                        if (Options.OutFormatEm)
                                            Header = Reader.ReadBytes(512);
                                        else if (Options.OutFormatMrc)
                                            Header = Reader.ReadBytes(1024);
                                    }

                                    try
                                    {
                                        if (SavedTemporary)
                                            File.Delete(Info.FullName);
                                    }
                                    catch { }
                                    Info = new FileInfo(Filename);
                                    
                                    //if (AsyncWriteTask != null)
                                        //AsyncWriteTask.Wait();
                                    //AsyncWriteTask = new Task(() =>
                                        {
                                            Log.Write("Writing results for " + Filename);

                                            bool Success = false;
                                            int FailCount = 0;
                                            uint OutputAverageLength = (uint)SubframeDims.Elements() / (uint)(Options.OutputDownsample * Options.OutputDownsample);
                                            uint OutputQuadAverageLength = (uint)Math.Max(1, Options.QuadSize * Options.QuadSize * Options.QuadsX * Options.QuadsY / (Options.OutputDownsample * Options.OutputDownsample));
                                            while(!Success)
                                            try
                                            {
                                                for (int r = 0; r < Options.NumberOutputRanges; r++)
                                                {
                                                    string RangeSuffix = Options.NumberOutputRanges == 1 ? "" : "f" + Options.OutputRangesArray[r].Start + "-" + Options.OutputRangesArray[r].End + ".";

                                                    using (BinaryWriter Writer = new BinaryWriter(File.Create(Options.OutputPath + Info.Name.Substring(0, Info.Name.Length - (FileExtension == "raw" ? "em" : FileExtension).Length) + RangeSuffix + OutputFileExtension)))
                                                    {
                                                        byte[] Buffer = null;
                                                        if(Options.OutFormatMrc && Options.OutFormatMRC16bit)
                                                            Buffer = new byte[OutputAverageLength * sizeof(short)];
                                                        else
                                                            Buffer = new byte[OutputAverageLength * sizeof(float)];
                                                        byte[] LocalHeader = new byte[Header.Length];
                                                        Array.Copy(Header, LocalHeader, Header.Length);
                                                        unsafe
                                                        {
                                                            fixed (byte* LocalHeaderPtr = LocalHeader)
                                                            {
                                                                if (Options.OutFormatMrc)
                                                                {
                                                                    ((int*)LocalHeaderPtr)[0] = FrameDims.X / Options.OutputDownsample;
                                                                    ((int*)LocalHeaderPtr)[1] = FrameDims.Y / Options.OutputDownsample;
                                                                    ((int*)LocalHeaderPtr)[2] = 1;
                                                                    ((int*)LocalHeaderPtr)[3] = Options.OutFormatMRC32bit ? 2 : 1;
                                                                }
                                                                else if (Options.OutFormatEm)
                                                                {
                                                                    LocalHeaderPtr[3] = (byte)5;
                                                                    ((int*)LocalHeaderPtr)[1] = FrameDims.X / Options.OutputDownsample;
                                                                    ((int*)LocalHeaderPtr)[2] = FrameDims.Y / Options.OutputDownsample;
                                                                    ((int*)LocalHeaderPtr)[3] = 1;
                                                                }
                                                            }
                                                            if (Options.OutFormatMrc && Options.OutFormatMRC16bit)
                                                            {
                                                                fixed (byte* BufferPtr = Buffer)
                                                                fixed (float* FramePtr = OutputAverage)
                                                                {
                                                                    short* BufferP = (short*)BufferPtr;
                                                                    float* FrameP = FramePtr + OutputAverageLength * r;
                                                                    for (int i = 0; i < OutputAverageLength; i++)
                                                                        *BufferP++ = (short)(*FrameP++ * 64f);
                                                                }
                                                            }
                                                            else
                                                            {
                                                                fixed (byte* BufferPtr = Buffer)
                                                                fixed (float* FramePtr = OutputAverage)
                                                                {
                                                                    float* BufferP = (float*)BufferPtr;
                                                                    float* FrameP = FramePtr + OutputAverageLength * r;
                                                                    for (int i = 0; i < OutputAverageLength; i++)
                                                                        *BufferP++ = *FrameP++;
                                                                }
                                                            }
                                                        }

                                                        Writer.Write(LocalHeader);
                                                        Writer.Write(Buffer);
                                                    }

                                                    for (int y = 0; y < Options.QuadsY; y++)
                                                        for (int x = 0; x < Options.QuadsX; x++)
                                                        {
                                                            int QuadLength = Options.QuadSize * Options.QuadSize / (Options.OutputDownsample * Options.OutputDownsample);
                                                            using (BinaryWriter Writer = new BinaryWriter(File.Create(Options.OutputPath + Info.Name.Substring(0, Info.Name.Length - (FileExtension == "raw" ? "em" : FileExtension).Length) + (x + 1) + "-" + (y + 1) + "." + RangeSuffix + OutputFileExtension)))
                                                            {
                                                                byte[] Buffer = null;
                                                                if (Options.OutFormatMrc && Options.OutFormatMRC16bit)
                                                                    Buffer = new byte[QuadLength * sizeof(short)];
                                                                else
                                                                    Buffer = new byte[QuadLength * sizeof(float)];

                                                                byte[] LocalHeader = new byte[Header.Length];
                                                                Array.Copy(Header, LocalHeader, Header.Length);
                                                                unsafe
                                                                {
                                                                    fixed (byte* LocalHeaderPtr = LocalHeader)
                                                                    {
                                                                        if (Options.OutFormatMrc)
                                                                        {
                                                                            ((int*)LocalHeaderPtr)[0] = Options.QuadSize / Options.OutputDownsample;
                                                                            ((int*)LocalHeaderPtr)[1] = Options.QuadSize / Options.OutputDownsample;
                                                                            ((int*)LocalHeaderPtr)[2] = 1;
                                                                            ((int*)LocalHeaderPtr)[3] = Options.OutFormatMRC32bit ? 2 : 1;
                                                                        }
                                                                        else if (Options.OutFormatEm)
                                                                        {
                                                                            LocalHeaderPtr[3] = (byte)5;
                                                                            ((int*)LocalHeaderPtr)[1] = Options.QuadSize / Options.OutputDownsample;
                                                                            ((int*)LocalHeaderPtr)[2] = Options.QuadSize / Options.OutputDownsample;
                                                                            ((int*)LocalHeaderPtr)[3] = 1;
                                                                        }
                                                                    }

                                                                    if (Options.OutFormatMrc && Options.OutFormatMRC16bit)
                                                                    {
                                                                        fixed (byte* BufferPtr = Buffer)
                                                                        fixed (float* FramePtr = OutputQuadAverages)
                                                                        {
                                                                            short* BufferP = (short*)BufferPtr;
                                                                            float* FrameP = FramePtr + OutputQuadAverageLength * r + QuadLength * (y * Options.QuadsX + x);
                                                                            for (int i = 0; i < QuadLength; i++)
                                                                                *BufferP++ = (short)(*FrameP++ * 64f);
                                                                        }
                                                                    }
                                                                    else
                                                                    {
                                                                        fixed (byte* BufferPtr = Buffer)
                                                                        fixed (float* FramePtr = OutputQuadAverages)
                                                                        {
                                                                            float* BufferP = (float*)BufferPtr;
                                                                            float* FrameP = FramePtr + OutputQuadAverageLength * r + QuadLength * (y * Options.QuadsX + x);
                                                                            for (int i = 0; i < QuadLength; i++)
                                                                                *BufferP++ = *FrameP++;
                                                                        }
                                                                    }
                                                                }

                                                                Writer.Write(LocalHeader);
                                                                Writer.Write(Buffer);
                                                            }
                                                        }
                                                }
                                                Success = true;
                                            }
                                            catch (Exception ex)
                                            {
                                                Log.Write(ex);
                                                Thread.Sleep(2000);
                                                FailCount++;
                                                if (FailCount % 50 == 0)
                                                {
                                                    string ErredPath = Options.OutputPath + Info.Name.Substring(0, Info.Name.Length - (FileExtension == "raw" ? "em" : FileExtension).Length);
                                                    MessageBox.Show("Failed to write output to " + ErredPath + " after 50 attempts. Please fix the problem and press OK to continue.");
                                                }
                                            }

                                            Log.Write("Results written for " + Filename);

                                            try
                                            {
                                                string[] DeletePatterns = Options.DeletePatterns.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                                                string[] DeleteFolders = Options.DeleteFolders.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                                                if (DeletePatterns.Length != DeleteFolders.Length)
                                                    return;

                                                for (int i = 0; i < DeletePatterns.Length; i++)
                                                {
                                                    foreach (string Path in Directory.EnumerateFiles(DeleteFolders[i], DeletePatterns[i], SearchOption.AllDirectories))
                                                        File.Delete(Path);

                                                    Log.Write("Deleted files matching " + DeletePatterns[i]);
                                                }
                                            }
                                            catch (Exception ex)
                                            {
                                                Log.Write(new Exception("Could not delete files matching pattern."));
                                                Log.Write(ex);
                                            }
                                        }//);
                                    //AsyncWriteTask.Start();

                                    if (AsyncArchiveTask != null)
                                        AsyncArchiveTask.Wait();
                                    AsyncArchiveTask = new Task(() =>
                                        {
                                            if (Options.ArchiveZip)
                                            {
                                                
                                                Log.Write("Compressing " + Filename);

                                                bool Success = false;
                                                int FailCount = 0;

                                                while(!Success)
                                                    try
                                                    {
                                                        using (BinaryReader Reader = new BinaryReader(File.OpenRead(Info.FullName)))
                                                        {
                                                            using (Ionic.BZip2.ParallelBZip2OutputStream Writer = new Ionic.BZip2.ParallelBZip2OutputStream(File.Create(Options.ArchivePath + Info.Name + ".bz2")))
                                                            {
                                                                Writer.MaxWorkers = 32;
                                                                long BytesToRead = Info.Length;
                                                                long BufferSize = 1024 * 1024 * 64;
                                                                Task WriterTask = null;
                                                                while (BytesToRead > 0)
                                                                {
                                                                    long CurrentBatch = Math.Min(BufferSize, BytesToRead);
                                                                    byte[] Buffer = new byte[CurrentBatch];
                                                                    Reader.Read(Buffer, 0, (int)CurrentBatch);
                                                                    if (WriterTask != null)
                                                                        WriterTask.Wait();
                                                                    WriterTask = Writer.WriteAsync(Buffer, 0, (int)CurrentBatch);
                                                                    BytesToRead -= CurrentBatch;
                                                                }
                                                                if (WriterTask != null)
                                                                    WriterTask.Wait();
                                                            }
                                                        }
                                                        Success = true;
                                                    }
                                                    catch (Exception ex)
                                                    {
                                                        Thread.Sleep(2000);
                                                        FailCount++;
                                                        if (FailCount % 10 == 0)
                                                        {
                                                            MessageBox.Show("Failed to write archive after 50 attempts. Please fix the problem and press OK to continue.");
                                                            Log.Write(ex);
                                                        }
                                                    }

                                                Log.Write("Compressed " + Filename);

                                                try
                                                {
                                                    if (SavedTemporary)
                                                        File.Delete(Info.FullName);
                                                }
                                                catch { }

                                                Log.Write("Deleted " + Filename + " (compressed copy saved to " + Options.ArchivePath + Info.Name + ".bz2)");
                                            }

                                            if (Options.ArchiveKeep)
                                            {
                                                //File.Move(Info.FullName, Options.ArchivePath + Info.Name);
                                            }

                                            ProgressIndicator.Dispatcher.Invoke(() => ProgressIndicator.Value++);
                                        });
                                    AsyncArchiveTask.Start();

                                    if (!IsRunning)
                                        break;

                                    ProcessedFiles.Add(Info.FullName);
                                    Thread.Sleep(50);
                                }
                                catch (Exception ex)
                                {
                                    Log.Write(ex);
                                }
                            }

                            if (Mode == ProcessingModes.Folder)
                            {
                                if (AsyncWriteTask != null)
                                    AsyncWriteTask.Wait();
                                if (AsyncArchiveTask != null)
                                    AsyncArchiveTask.Wait();
                            }

                            ProgressIndicator.Dispatcher.Invoke(() =>
                            {
                                ProgressIndicator.Visibility = Visibility.Hidden;
                            });

                            if (Mode == ProcessingModes.Folder)
                                IsRunning = false;
                        }
                                        
                    if (AsyncWriteTask != null)
                        AsyncWriteTask.Wait();
                    if (AsyncArchiveTask != null)
                        AsyncArchiveTask.Wait();

                    ButtonProcess.Dispatcher.Invoke(() =>
                        {
                            ButtonProcess.Content = "START";
                            ButtonProcess.IsEnabled = true;
                            StackOptions.IsEnabled = true;
                            ProgressIndicator.Visibility = Visibility.Hidden;
                        });
                });

            IsRunning = true;
            ProcessingTask.Start();
            ButtonProcess.Content = "CANCEL";
            StackOptions.IsEnabled = false;
        }

        private void ButtonMoreOutputRanges_Click(object sender, RoutedEventArgs e)
        {
            PanelOutputRanges_MouseLeave(null, null);
            PanelOutputRangeSliders.Visibility = Visibility.Collapsed;
            PanelOutputRangeText.Visibility = Visibility.Visible;
            TextOutputRanges.TriggerEdit();
        }

        private void PanelOutputRanges_MouseEnter(object sender, MouseEventArgs e)
        {
            if (Options.NumberOutputRanges <= 1)
            {
                ShadowOutputRanges.BlurRadius = 5;
                PanelOutputRanges.Background = new SolidColorBrush(Colors.White);
                ButtonMoreOutputRanges.Visibility = Visibility.Visible;
            }
        }

        private void PanelOutputRanges_MouseLeave(object sender, MouseEventArgs e)
        {
            if (Options.NumberOutputRanges <= 1)
            {
                ShadowOutputRanges.BlurRadius = 0;
                PanelOutputRanges.Background = new SolidColorBrush(Colors.Transparent);
                ButtonMoreOutputRanges.Visibility = Visibility.Collapsed;
            }
        }

        private void PanelMRCOptions_MouseEnter(object sender, MouseEventArgs e)
        {
            ShadowMRCOptions.BlurRadius = 5;
            PanelMRCOptions.Background = new SolidColorBrush(Colors.White);
            PanelMRCOptionsBits.Visibility = Visibility.Visible;
        }

        private void PanelMRCOptions_MouseLeave(object sender, MouseEventArgs e)
        {
            ShadowMRCOptions.BlurRadius = 0;
            PanelMRCOptions.Background = new SolidColorBrush(Colors.Transparent);
            PanelMRCOptionsBits.Visibility = Visibility.Collapsed;
        }
    }

    public enum ProcessingModes
    { 
        Folder = 1,
        Daemon = 2
    }
}
