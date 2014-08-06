using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace K2AlignDaemon
{
    public class ProcessingOptions : Sparta.DataBase
    {
        private string _InputPath = "";
        public string InputPath
        {
            get { return _InputPath; }
            set { if (value != _InputPath) { _InputPath = value; OnPropertyChanged(); } }
        }

        private string _OutputPath = "";
        public string OutputPath
        {
            get { return _OutputPath; }
            set { if (value != _OutputPath) { _OutputPath = value; OnPropertyChanged(); } }
        }

        private string _ArchivePath = "";
        public string ArchivePath
        {
            get { return _ArchivePath; }
            set { if (value != _ArchivePath) { _ArchivePath = value; OnPropertyChanged(); } }
        }

        private bool _InFormatRaw = true;
        public bool InFormatRaw
        {
            get { return _InFormatRaw; }
            set { if (value != _InFormatRaw) { _InFormatRaw = value; OnPropertyChanged(); } }
        }

        private bool _InFormatFeiRaw = false;
        public bool InFormatFeiRaw
        {
            get { return _InFormatFeiRaw; }
            set { if (value != _InFormatFeiRaw) { _InFormatFeiRaw = value; OnPropertyChanged(); } }
        }

        private bool _InFormatMrc = false;
        public bool InFormatMrc
        {
            get { return _InFormatMrc; }
            set { if (value != _InFormatMrc) { _InFormatMrc = value; OnPropertyChanged(); } }
        }

        private bool _InFormatEm = false;
        public bool InFormatEm
        {
            get { return _InFormatEm; }
            set { if (value != _InFormatEm) { _InFormatEm = value; OnPropertyChanged(); } }
        }

        private bool _OutFormatMrc = true;
        public bool OutFormatMrc
        {
            get { return _OutFormatMrc; }
            set { if (value != _OutFormatMrc) { _OutFormatMrc = value; OnPropertyChanged(); } }
        }

        private bool _OutFormatEm = false;
        public bool OutFormatEm
        {
            get { return _OutFormatEm; }
            set { if (value != _OutFormatEm) { _OutFormatEm = value; OnPropertyChanged(); } }
        }

        private bool _OutFormatMRC16bit = false;
        public bool OutFormatMRC16bit
        {
            get { return _OutFormatMRC16bit; }
            set { if (value != _OutFormatMRC16bit) { _OutFormatMRC16bit = value; OnPropertyChanged(); } }
        }

        private bool _OutFormatMRC32bit = true;
        public bool OutFormatMRC32bit
        {
            get { return _OutFormatMRC32bit; }
            set { if (value != _OutFormatMRC32bit) { _OutFormatMRC32bit = value; OnPropertyChanged(); } }
        }

        private int _RawWidth = 7676;
        public int RawWidth
        {
            get { return _RawWidth; }
            set { if (value != _RawWidth) { _RawWidth = value; OnPropertyChanged(); } }
        }

        private int _RawHeight = 7420;
        public int RawHeight
        {
            get { return _RawHeight; }
            set { if (value != _RawHeight) { _RawHeight = value; OnPropertyChanged(); } }
        }

        private int _RawDepth = 10;
        public int RawDepth
        {
            get { return _RawDepth; }
            set 
            { 
                if (value != _RawDepth) 
                { 
                    _RawDepth = value;

                    MinValidFrames = Math.Min(MinValidFrames, RawDepth);
                    ProcessLast = Math.Min(ProcessLast, RawDepth);

                    OnPropertyChanged(); 
                } 
            }
        }

        private int _Width = 7420;
        public int Width
        {
            get { return _Width; }
            set { if (value != _Width) { _Width = value; OnPropertyChanged(); } }
        }

        private int _Height = 7676;
        public int Height
        {
            get { return _Height; }
            set { if (value != _Height) { _Height = value; OnPropertyChanged(); } }
        }

        private int _Depth = 10;
        public int Depth
        {
            get { return _Depth; }
            set { if (value != _Depth) { _Depth = value; OnPropertyChanged(); } }
        }

        private bool _ArchiveZip = true;
        public bool ArchiveZip
        {
            get { return _ArchiveZip; }
            set { if (value != _ArchiveZip) { _ArchiveZip = value; OnPropertyChanged(); } }
        }

        private bool _ArchiveKeep = false;
        public bool ArchiveKeep
        {
            get { return _ArchiveKeep; }
            set { if (value != _ArchiveKeep) { _ArchiveKeep = value; OnPropertyChanged(); } }
        }

        private bool _CorrectGain = true;
        public bool CorrectGain
        {
            get { return _CorrectGain; }
            set { if (value != _CorrectGain) { _CorrectGain = value; OnPropertyChanged(); } }
        }

        private string _GainPath = "";
        public string GainPath
        {
            get { return _GainPath; }
            set { if (value != _GainPath) { _GainPath = value; OnPropertyChanged(); } }
        }

        public bool CustomGain = false;

        private bool _CorrectXray = true;
        public bool CorrectXray
        {
            get { return _CorrectXray; }
            set { if (value != _CorrectXray) { _CorrectXray = value; OnPropertyChanged(); } }
        }

        private decimal _BandpassLow = 0.001M;
        public decimal BandpassLow
        {
            get { return _BandpassLow; }
            set { if (value != _BandpassLow) { _BandpassLow = value; OnPropertyChanged(); } }
        }

        private decimal _BandpassHigh = 0.063M;
        public decimal BandpassHigh
        {
            get { return _BandpassHigh; }
            set { if (value != _BandpassHigh) { _BandpassHigh = value; OnPropertyChanged(); } }
        }

        private int _ProcessFirst = 1;
        public int ProcessFirst
        {
            get { return _ProcessFirst; }
            set 
            { 
                if (value != _ProcessFirst) 
                { 
                    _ProcessFirst = value;

                    OutputFirst = Math.Max(OutputFirst, ProcessFirst);

                    OnPropertyChanged(); 
                } 
            }
        }

        private int _ProcessLast = 10;
        public int ProcessLast
        {
            get { return _ProcessLast; }
            set 
            { 
                if (value != _ProcessLast) 
                { 
                    _ProcessLast = value;

                    OutputLast = Math.Min(OutputLast, ProcessLast);

                    OnPropertyChanged(); 
                } 
            }
        }

        private int _OutputFirst = 1;
        public int OutputFirst
        {
            get { return _OutputFirst; }
            set 
            { 
                if (value != _OutputFirst) 
                { 
                    _OutputFirst = value;

                    ProcessFirst = Math.Min(ProcessFirst, OutputFirst);
                    FrameRange[] Ranges = OutputRangesArray;
                    Ranges[0].Start = value;
                    SetOutputRanges(Ranges);

                    OnPropertyChanged(); 
                } 
            }
        }

        private int _OutputLast = 10;
        public int OutputLast
        {
            get { return _OutputLast; }
            set 
            { 
                if (value != _OutputLast) 
                { 
                    _OutputLast = value;

                    ProcessLast = Math.Max(ProcessLast, OutputLast);
                    FrameRange[] Ranges = OutputRangesArray;
                    Ranges[0].End = value;
                    SetOutputRanges(Ranges);

                    OnPropertyChanged(); 
                } 
            }
        }

        private string _OutputRanges = "1-10";
        public string OutputRanges
        {
            get { return _OutputRanges; }
            set 
            { 
                if (value != _OutputRanges) 
                {
                    bool IsValid = false;
                    try
                    {
                        FrameRange[] Test = OutputRangesArray;
                        IsValid = true;
                    }
                    catch
                    {
                        value = (new FrameRange(_OutputFirst, _OutputLast)).ToString();
                    }

                    _OutputRanges = value;

                    FrameRange[] Ranges = OutputRangesArray;
                    OutputFirst = Ranges[0].Start;
                    OutputLast = Ranges[0].End;

                    OnPropertyChanged(); 
                } 
            }
        }
        public void SetOutputRanges(FrameRange[] ranges)
        {
            string[] Parts = new string[ranges.Length];
            for (int i = 0; i < ranges.Length; i++)
                Parts[i] = ranges[i].ToString();

            OutputRanges = string.Join(";", Parts);
        }

        public FrameRange[] OutputRangesArray
        {
            get
            {
                string[] Parts = _OutputRanges.Replace(" ", "").Replace(",", ";").Split(new char[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
                FrameRange[] Ranges = new FrameRange[Parts.Length];
                for (int i = 0; i < Parts.Length; i++)
                    Ranges[i] = FrameRange.Parse(Parts[i]);

                return Ranges;
            }
        }

        public int[] OutputRangesIntArray
        {
            get
            {
                List<int> IntRanges = new List<int>();
                foreach (var Range in OutputRangesArray)
                {
                    IntRanges.Add(Range.Start - 1);
                    IntRanges.Add(Range.End - 1);
                }

                return IntRanges.ToArray();
            }
        }

        public int NumberOutputRanges
        {
            get
            {
                return OutputRangesArray.Length;
            }
        }

        private decimal _MaxDrift = 100M;
        public decimal MaxDrift
        {
            get { return _MaxDrift; }
            set { if (value != _MaxDrift) { _MaxDrift = value; OnPropertyChanged(); } }
        }

        private int _MinValidFrames = 2;
        public int MinValidFrames
        {
            get { return _MinValidFrames; }
            set 
            { 
                if (value != _MinValidFrames) 
                { 
                    _MinValidFrames = value;

                    RawDepth = Math.Max(RawDepth, MinValidFrames);

                    OnPropertyChanged(); 
                } 
            }
        }

        private int _AverageWindow = 1;
        public int AverageWindow
        {
            get { return _AverageWindow; }
            set 
            { 
                if (value != _AverageWindow) 
                { 
                    _AverageWindow = value;

                    //ProcessLast = Math.Max(ProcessLast, ProcessFirst + value - 1);
                    //OutputLast = Math.Max(OutputLast, OutputFirst + value - 1);
                    //OutputFirst = Math.Max(OutputFirst, Math.Max((value - 1) / 2 + 1, 1));
                    //OutputLast = Math.Min(OutputLast, Math.Max(ProcessLast - (value - 1) / 2, 1));

                    OnPropertyChanged(); 
                } 
            }
        }

        private int _QuadsX = 0;
        public int QuadsX
        {
            get { return _QuadsX; }
            set { if (value != _QuadsX) { _QuadsX = value; OnPropertyChanged(); } }
        }

        private int _QuadsY = 0;
        public int QuadsY
        {
            get { return _QuadsY; }
            set { if (value != _QuadsY) { _QuadsY = value; OnPropertyChanged(); } }
        }

        private int _QuadSize = 4096;
        public int QuadSize
        {
            get { return _QuadSize; }
            set { if (value != _QuadSize) { _QuadSize = value; OnPropertyChanged(); } }
        }

        private int _OutputDownsample = 1;
        public int OutputDownsample
        {
            get { return _OutputDownsample; }
            set { if (value != _OutputDownsample) { _OutputDownsample = value; OnPropertyChanged(); } }
        }

        private string _DeletePatterns = "";
        public string DeletePatterns
        {
            get { return _DeletePatterns; }
            set { if (value != _DeletePatterns) { _DeletePatterns = value; OnPropertyChanged(); } }
        }

        private string _DeleteFolders = "";
        public string DeleteFolders
        {
            get { return _DeleteFolders; }
            set { if (value != _DeleteFolders) { _DeleteFolders = value; OnPropertyChanged(); } }
        }
        
        public void TryFindMask()
        { 
            
        }
    }

    public class FrameRange
    {
        public int Start, End;

        public FrameRange(int start, int end)
        {
            Start = start;
            End = end;
        }

        public override string ToString()
        {
            return Start + "-" + End;
        }

        public static FrameRange Parse(string literal)
        {
            string[] Parts = literal.Split(new char[] { '-' }, StringSplitOptions.RemoveEmptyEntries);
            if (Parts.Length != 2)
                throw new Exception();

            return new FrameRange(int.Parse(Parts[0]), int.Parse(Parts[1]));
        }
    }
}
