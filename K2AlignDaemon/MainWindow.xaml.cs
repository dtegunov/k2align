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

        /// <summary>
        /// Upon start, create temp folder if there is none. If it contains files, ask user if those should be deleted.
        /// Populate the Options class with previously saved values.
        /// Open log file, start session.
        /// </summary>
        public MainWindow()
        {
            InitializeComponent();
            Closing += MainWindow_Closing;
            Options.PropertyChanged += Options_PropertyChanged;

            if (!Directory.Exists("temp"))
                Directory.CreateDirectory("temp");

            List<string> TempFiles = new List<string>();
            foreach (var Path in Directory.EnumerateFiles("temp"))
                TempFiles.Add(Path);
            if (TempFiles.Count > 0)
            { 
                if(MessageBox.Show("There are some files in /temp. Delete them?", "Maintenance", MessageBoxButton.YesNo) == MessageBoxResult.Yes)
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
            Options.OutputStack = Properties.Settings.Default.OutputStack;

            Log = new LogWriter("log.txt");
            Log.Write(string.Format("Session started by {0}", System.Security.Principal.WindowsIdentity.GetCurrent().Name));
        }

        /// <summary>
        /// Handle selection of multiple output frame ranges.
        /// </summary>
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

        /// <summary>
        /// Before the application closes, save all options and close log file.
        /// </summary>
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
            Properties.Settings.Default.OutputStack = Options.OutputStack;

            Properties.Settings.Default.Save();
            Log.Dispose();
        }

        /// <summary>
        /// Folder processing has been selected on the start screen.
        /// Show folder selection dialog.
        /// Once selected, try to open every file to check permissions. If this fails, let the user select another folder.
        /// When everything is OK, proceed to options screen.
        /// </summary>
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
                    this.Height = 770;
                    if (!Options.ArchiveZip)
                        this.Height -= 30;
                    if (!Options.InFormatRaw)
                        this.Height -= 20;

                    Log.Write("Doing folder mode in " + Dialog.SelectedPath);
                    break;
                }
                else
                    break;
        }

        /// <summary>
        /// On-the-fly processing has been selected on the start screen.
        /// Show folder selection dialog to determine where new frame stacks will come in during acquisition.
        /// Once selected, try to open every file to check permissions. If this fails, let the user select another folder.
        /// When everything is OK, proceed to the options screen.
        /// </summary>
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
                    this.Height = 830;
                    if (!Options.ArchiveZip)
                        this.Height -= 30;
                    if (!Options.InFormatRaw)
                        this.Height -= 20;

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
            
            //For each of the 4 supported formats, check if files with that extension are present in folder.
            //If yes, open one of them, read the header and adjust the settings.
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
                //This is the raw, header-less format produced by TOM acquisition
                //Default dimensions are 7676x7420

                Options.InFormatMrc = false;
                Options.InFormatEm = false;
                Options.InFormatRaw = true;

                FrameDims = new int3(7676, 7420, Options.RawDepth);
            }
            else if (Directory.EnumerateFiles(Options.InputPath, "*.raw").Count() > 0)
            {
                //This is Falcon II/EPU's format. It does have a 47 byte long header that includes dimensions, but those have been 4096² so far.

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

            //If the determined resolution matches any of the provided gain masks, select one
            if (FrameDims.X == 7676 && FrameDims.Y == 7420)
                Options.GainPath = "gain_7676x7420.em";
            else if (FrameDims.Y == 7676 && FrameDims.X == 7420)
                Options.GainPath = "gain_7420x7676.em";
            if (FrameDims.X == 3838 && FrameDims.Y == 3710)
                Options.GainPath = "gain_3838x3710.em";
            else if (FrameDims.Y == 3838 && FrameDims.X == 3710)
                Options.GainPath = "gain_3710x3838.em";
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
            this.Height += 30;
        }

        private void RadioArchiveZip_Unchecked(object sender, RoutedEventArgs e)
        {
            GridTransferPath.Visibility = Visibility.Collapsed;
            this.Height -= 30;
        }

        private void RadioInFormatRaw_Checked(object sender, RoutedEventArgs e)
        {
            StackRawDims.Visibility = Visibility.Visible;
            this.Height += 20;
        }

        private void RadioInFormatRaw_Unchecked(object sender, RoutedEventArgs e)
        {
            StackRawDims.Visibility = Visibility.Collapsed;
            this.Height -= 20;
        }

        /// <summary>
        /// Show folder selection dialog to determine where aligned output files should be written.
        /// After selection, check write permissions for the folder and notify user on error.
        /// </summary>
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

        /// <summary>
        /// Show folder selection dialog to determine where input files should be archived.
        /// After selection, check write permissions for the folder and notify user on error.
        /// </summary>
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

        /// <summary>
        /// Show file selection dialog to choose a gain reference in EM format.
        /// </summary>
        private void ButtonGainpath_Click(object sender, RoutedEventArgs e)
        {
            System.Windows.Forms.OpenFileDialog Dialog = new System.Windows.Forms.OpenFileDialog();
            Dialog.Filter = "Image Files|*.em;*.mrc";
            Dialog.Multiselect = false;
            System.Windows.Forms.DialogResult Result = Dialog.ShowDialog();

            if (Result.ToString() == "OK")
            {
                Options.GainPath = Dialog.FileName;
                Options.CustomGain = true;
            }
        }

        /// <summary>
        /// The main alignment procedure. Button functions both as start and stop command depending on current state.
        /// </summary>
        private void ButtonProcess_Click(object sender, RoutedEventArgs e)
        {
            //If alignment is running, tell the processing thread to stop by setting IsRunning to false.
            //The processing thread will then finish processing the current file and finally stop.
            //While this last file is being processed, starting new alignment is not possible.
            if (IsRunning)
            {
                ButtonProcess.Content = "STOPPING...";
                ButtonProcess.IsEnabled = false;
                IsRunning = false;
                return;
            }

            //The function executed for processing is defined as a Lambda.
            //Within it, two additional lambdas are defined for archiving and writing the output, and started asynchroniously.
            //The processing task is startet at the bottom of ButtonProcess_Click
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

                    //Make sure folder paths end in a uniform way, i. e. with \
                    if (Options.InputPath[Options.InputPath.Length - 1] != '\\' && Options.InputPath[Options.InputPath.Length - 1] != '/')
                        Options.InputPath += '\\';
                    if(Options.ArchiveZip)
                        if (Options.ArchivePath[Options.ArchivePath.Length - 1] != '\\' && Options.ArchivePath[Options.ArchivePath.Length - 1] != '/')
                            Options.ArchivePath += '\\';
                    
                    //If gain correction is used, read the reference
                    int3 GainDims = new int3(1, 1, 1);
                    float[] GainMask = new float[1];
                    if(Options.CorrectGain)
                    {
                        GainDims = IOHelper.GetMapDimensions(Options.GainPath);
                        GainMask = IOHelper.ReadMapFloat(Options.GainPath);
                    }

                    //corrupt.txt is used to save file names and frame numbers when black squares are detected (K2 issue)
                    using(TextWriter CorruptWriter = new StreamWriter(File.Create(Options.OutputPath + "corrupt.txt")))
                        //Loop runs until stopped for on-the-fly, or one time for folder processing
                        while (IsRunning)
                        {
                            //Find out if the RAW format selected is from TOM or EPU.
                            //If it is EPU, sub-frames will come in separate files named _n0 to _n6 for Falcon II's 7 sub-frame bursts.
                            if (Options.InFormatRaw && Directory.EnumerateFiles(Options.InputPath, "*_n0.raw").Count() > 0)
                            {
                                FileExtension = "raw";
                                FileWildcard = "*_n0";
                            }

                            IEnumerable<string> VolatileFiles = Directory.EnumerateFiles(Options.InputPath, FileWildcard + "." + FileExtension);
                            Dictionary<string, long> FileSizes = new Dictionary<string, long>();
                            List<string> Files = new List<string>();

                            //Additional processing for the separate files produced by EPU.
                            //One Falcon II burst consists of 7 frames. Depending on the specified sub-frame range,
                            //multiple bursts may be needed. Then they constitute a group.
                            //The loop waits for parts of the current group to arrive in the folder and saves their names.
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

                                    //Determine how many 7 sub-frame bursts belong to the same group.
                                    int CombinedFilesNeeded = (Options.ProcessLast - Options.ProcessFirst + 1 + 6) / 7;
                                    List<string> CombinedNames = new List<string>();
                                    DateTime GroupStartDate = new DateTime(1950, 1, 1);

                                    //While burst group has not been completed
                                    while (CombinedNames.Count < CombinedFilesNeeded)
                                    {
                                        if (!IsRunning)
                                            break;

                                        //VolatileFiles contains paths to all *_n0.raw currently present in the folder.
                                        foreach (string Filename in VolatileFiles)
                                        {
                                            if (ProcessedFiles.Contains(Filename))
                                                continue;
                                            ProcessedFiles.Add(Filename);

                                            //Bursts are assumed to belong to the same group if the time difference between 
                                            //two consecutive bursts is less than 28 seconds.

                                            FileInfo Info = new FileInfo(Filename);
                                            string DatePart = Info.Name.Substring(Info.Name.IndexOf('_') + 1, 15);
                                            DateTime FileDate = DateTime.ParseExact(DatePart, "yyyyMMdd_HHmmss", CultureInfo.InvariantCulture.DateTimeFormat);
                                            TimeSpan Difference = FileDate - GroupStartDate;

                                            //If time difference is more than 28 seconds and the group is still not filled,
                                            //dismiss previous group members and start filling from scratch.
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

                                    //The name root of the first burst in a group is used for naming the stack.
                                    FileInfo RootInfo = new FileInfo(CombinedNames[0]);
                                    string NameRoot = RootInfo.Name.Substring(0, RootInfo.Name.IndexOf("_n0.raw"));

                                    //Combine all files in group into an EM stack.
                                    using (BinaryWriter Writer = new BinaryWriter(File.Create("temp\\" + NameRoot + ".em")))
                                    {
                                        //Write EM header.
                                        HeaderEM Header = new HeaderEM();
                                        Header.Dimensions = new int3(Options.RawWidth, Options.RawHeight, CombinedFilesNeeded * 7);
                                        Header.Mode = EMDataType.Long;
                                        Header.Write(Writer);

                                        //For each burst in group:
                                        foreach (string Filename in CombinedNames)
                                        {
                                            FileInfo Info = new FileInfo(Filename);
                                            string LocalNameRoot = Info.Name.Substring(0, RootInfo.Name.IndexOf("_n0.raw"));
                                            string LastFrameName = LocalNameRoot + "_n6.raw";

                                            //Ensure the burst is completely written out on the EPU machine, i. e. _n6.raw is there.
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

                                                //Attempt to read sub-frame and handle exception in case of network problems.
                                                //Read data is written to the open EM stack.
                                                while (!Success)
                                                {
                                                    try
                                                    {
                                                        File.Copy(Options.InputPath + FrameName, "temp\\" + FrameName);
                                                        using (BinaryReader Reader = new BinaryReader(File.OpenRead("temp\\" + FrameName)))
                                                        {
                                                            Reader.ReadBytes(49);
                                                            byte[] Buffer = Reader.ReadBytes(Options.RawWidth * Options.RawHeight * sizeof(int));
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

                                                Success = false;
                                                FailCount = 0;
                                                //Attempt to delete temporary copy of the sub-frame raw.
                                                while (!Success)
                                                {
                                                    try
                                                    {
                                                        File.Delete("temp\\" + FrameName);
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
                                        Log.Write("Created temp\\" + NameRoot + ".em");
                                    }

                                    //Add newly created EM stack to the list of files to be processed.
                                    Files.Add("temp\\" + NameRoot + ".em");

                                    //In on-the-fly mode, don't wait for other files to be combined, process this one.
                                    if (Mode == ProcessingModes.Daemon)
                                        break;
                                }
                            }
                            //Everything else is easier than EPU handling.
                            else
                            {
                                //Add current size of each file in folder to monitor if it is still being written out...
                                foreach (string Filename in VolatileFiles)
                                {
                                    FileInfo Info = new FileInfo(Filename);
                                    FileSizes.Add(Filename, Info.Length);
                                }
                            }

                            //... wait one second...
                            if (Mode == ProcessingModes.Daemon)
                                Thread.Sleep(1000);

                            //... and check if the size stayed the same. If yes, file can be processed.
                            foreach (string Filename in VolatileFiles)
                            {
                                FileInfo Info = new FileInfo(Filename);
                                if (FileSizes.ContainsKey(Filename) && Info.Length == FileSizes[Filename])
                                    Files.Add(Filename);
                            }

                            //Tell GUI thread to show progress indicator.
                            ProgressIndicator.Dispatcher.Invoke(() => 
                                {
                                    ProgressIndicator.Visibility = Visibility.Visible;
                                    ProgressIndicator.Value = 0;
                                    ProgressIndicator.Maximum = Files.Count();
                                });

                            //Files to be processed contain a single EM stack in case of EPU RAW,
                            //every file in folder not noticed to change size in other case of on-the-fly,
                            //or the entire folder contents in case of folder processing.
                            foreach (string Filename in Files)
                            {
                                try
                                {
                                    FileInfo Info = new FileInfo(Filename);
                                    bool SavedTemporary = false;

                                    //If file is not on local C drive, copy it to temp.
                                    //For some reason, this is faster than directly reading it from the network.
                                    if (Info.FullName[0].ToString().ToLower() != "c")
                                    {
                                        File.Copy(Info.FullName, "temp\\" + Info.Name, true);
                                        SavedTemporary = true;
                                        Info = new FileInfo("temp\\" + Info.Name);
                                    }

                                    int RawDataType = 1;
                                    if (ProcessedFiles.Contains(Info.FullName))
                                        continue;
                                    int3 FrameDims = new int3(1, 1, 1);

                                    if (Options.InFormatMrc || Options.InFormatEm || FileExtension == "raw")
                                        FrameDims = IOHelper.GetMapDimensions(Info.FullName);
                                    else if (Options.InFormatRaw)
                                        FrameDims = new int3(Options.RawWidth, Options.RawHeight, Options.RawDepth);

                                    //TOM's RAW can come in two different flavors: uchar and float.
                                    //In the same place, check if file size matches specified dimensions. If not, do not process.
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

                                    //Allocate memory for alignment output, both whole frame and quads.
                                    int3 SubframeDims = new int3(FrameDims.X, FrameDims.Y, 1);
                                    bool[] IsCorrupt = new bool[] { false };
                                    float[] OutputAverage = new float[(uint)SubframeDims.Elements() / (Options.OutputDownsample * Options.OutputDownsample) * Options.NumberOutputRanges];
                                    float[] OutputQuadAverages = new float[Math.Max(1, Options.QuadSize * Options.QuadSize * Options.QuadsX * Options.QuadsY / (Options.OutputDownsample * Options.OutputDownsample)) * Options.NumberOutputRanges];

                                    //Finally, align!
                                    GPU.FrameAlign(Info.FullName,
                                                   OutputAverage,
                                                   OutputQuadAverages,
                                                   Options.CorrectGain,
                                                   GainMask,
                                                   GainDims,
                                                   Options.CorrectXray,
                                                   (float)Options.BandpassLow / 2f,
                                                   (float)Options.BandpassHigh / 2f,
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
                                                   Options.OutputStack,
                                                   Options.OutputPath + Info.Name.Substring(0, Info.Name.Length - (FileExtension == "raw" ? "em" : FileExtension).Length - 1),
                                                   IsCorrupt,
                                                   Options.OutputPath + Info.Name + ".log");

                                    //Black squares detected.
                                    if (IsCorrupt[0])
                                        CorruptWriter.WriteLine(Info.Name);

                                    //If input and output are same format, copy header from input and update dimensions.
                                    //Otherwise create new header.
                                    MapHeader Header = null;
                                    if (FileExtension == OutputFileExtension)
                                        Header = MapHeader.ReadFromFile(Info.FullName);
                                    else if (OutputFileExtension == "em")
                                        Header = new HeaderEM();
                                    else if (OutputFileExtension == "mrc")
                                        Header = new HeaderMRC();
                                    else
                                        throw new Exception("Could not create header because format is not supported.");

                                    if (Options.OutFormatMRC16bit)
                                        Header.SetValueType(typeof(ushort));
                                    else
                                        Header.SetValueType(typeof(float));

                                    //If something was saved as local temporary copy, delete it.
                                    //Unless it still has to be zipped.
                                    try
                                    {
                                        if ((SavedTemporary || Info.FullName.ToLower().Contains("temp")) && !Options.ArchiveZip)
                                        {
                                            File.Delete(Info.FullName);
                                            Info = new FileInfo(Filename);
                                            SavedTemporary = false;
                                        }
                                    }
                                    catch { }
                                    finally 
                                    {
                                        Info = new FileInfo(Filename);
                                    }
                                    
                                    //Asynchronously write averages to output folder.
                                    if (AsyncWriteTask != null)
                                        AsyncWriteTask.Wait();
                                    AsyncWriteTask = new Task(() =>
                                    {
                                        //Write the output of whole frame and quads.

                                        Log.Write("Writing results for " + Filename);

                                        bool Success = false;
                                        int FailCount = 0;
                                        uint OutputAverageLength = (uint)SubframeDims.Elements() / (uint)(Options.OutputDownsample * Options.OutputDownsample);
                                        uint OutputQuadAverageLength = (uint)Math.Max(1, Options.QuadSize * Options.QuadSize * Options.QuadsX * Options.QuadsY / (Options.OutputDownsample * Options.OutputDownsample));

                                        //Attempt to write, throw up after 50 fails.
                                        while (!Success)
                                            try
                                            {
                                                //Each output frame range has its own output.
                                                for (int r = 0; r < Options.NumberOutputRanges; r++)
                                                {
                                                    string RangeSuffix = Options.NumberOutputRanges == 1 ? "" : "f" + Options.OutputRangesArray[r].Start + "-" + Options.OutputRangesArray[r].End + ".";

                                                    Header.Dimensions = new int3(FrameDims.X / Options.OutputDownsample,
                                                                                    FrameDims.Y / Options.OutputDownsample,
                                                                                    1);
                                                    float[] WholeBuffer = new float[Header.Dimensions.Elements()];
                                                    Array.Copy(OutputAverage, WholeBuffer.Length * r, WholeBuffer, 0, WholeBuffer.Length);
                                                    //Write aligned whole frame.
                                                    IOHelper.WriteMapFloat(Options.OutputPath + Info.Name.Substring(0, Info.Name.Length - (FileExtension == "raw" ? "em" : FileExtension).Length) + RangeSuffix + OutputFileExtension,
                                                                           Header,
                                                                           WholeBuffer);

                                                    //Write each aligned quad.
                                                    for (int y = 0; y < Options.QuadsY; y++)
                                                        for (int x = 0; x < Options.QuadsX; x++)
                                                        {
                                                            int QuadLength = Options.QuadSize * Options.QuadSize / (Options.OutputDownsample * Options.OutputDownsample);
                                                            float[] Buffer = new float[QuadLength];
                                                            Array.Copy(OutputQuadAverages, OutputQuadAverageLength * r + QuadLength * (y * Options.QuadsX + x), Buffer, 0, QuadLength);
                                                            Header.Dimensions = new int3(Options.QuadSize / Options.OutputDownsample,
                                                                                            Options.QuadSize / Options.OutputDownsample,
                                                                                            1);

                                                            IOHelper.WriteMapFloat(Options.OutputPath + Info.Name.Substring(0, Info.Name.Length - (FileExtension == "raw" ? "em" : FileExtension).Length) + (x + 1) + "-" + (y + 1) + "." + RangeSuffix + OutputFileExtension,
                                                                                    Header,
                                                                                    Buffer);
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

                                        //Attempt to delete temp files on EPU machine, as specified by folder and name pattern.
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
                                    });
                                    AsyncWriteTask.Start();

                                    if (AsyncArchiveTask != null)
                                        AsyncArchiveTask.Wait();
                                    //While the next frames are processed, archive this one in parallel, if needed.
                                    AsyncArchiveTask = new Task(() =>
                                        {
                                            //Zip is done using bzip2, utilizing up to 32 threads.
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
                                                            //Ionic provides a bzip2 file stream that makes compressing as easy as writing a regular file.
                                                            //Read and compress/write is done in parallel using a small (64 MB) intermediate buffer.
                                                            using (Ionic.BZip2.ParallelBZip2OutputStream Writer = new Ionic.BZip2.ParallelBZip2OutputStream(File.Create(Options.ArchivePath + Info.Name + ".bz2")))
                                                            {
                                                                Writer.MaxWorkers = 32;
                                                                long BytesToRead = Info.Length;
                                                                long BufferSize = 1024 * 1024 * 64; //Empirically determined to maximize speed
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
                                                    if (SavedTemporary || Info.FullName.ToLower().Contains("temp"))
                                                        File.Delete(Info.FullName);
                                                }
                                                catch { }

                                                Log.Write("Deleted " + Filename + " (compressed copy saved to " + Options.ArchivePath + Info.Name + ".bz2)");
                                            }

                                            //Tell GUI thread to update progess indicator.
                                            ProgressIndicator.Dispatcher.Invoke(() => ProgressIndicator.Value++);
                                        });
                                    AsyncArchiveTask.Start();

                                    if (!IsRunning)
                                        break;

                                    ProcessedFiles.Add(Filename);
                                    Thread.Sleep(50);
                                }
                                catch (Exception ex)
                                {
                                    Log.Write(ex);
                                }
                            }

                            //In folder processing mode, there is only one iteration of the main loop, so make sure everything is finished.
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
