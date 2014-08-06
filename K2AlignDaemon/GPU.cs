using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Threading;

namespace K2AlignDaemon
{
    public static class GPU
    {
        [DllImport("K2Align.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "h_FrameAlign")]
        public static extern void FrameAlign([MarshalAs(UnmanagedType.AnsiBStr)] string imagepath, float[] h_outputwhole, float[] h_outputquads,
                                            bool correctgain, float[] h_gainfactor, int3 gainfactordims,
                                            bool correctxray,
                                            float bandpasslow, float bandpasshigh,
                                            [MarshalAs(UnmanagedType.AnsiBStr)] string subframeformat, int rawdatatype,
                                            int3 rawdims,
                                            int averageextent,
                                            int adjacentgap,
                                            int firstframe, int lastframe,
                                            int[] h_outputranges, int numberoutputranges,
                                            float maxdrift, int minvalidframes,
                                            int outputdownsamplefactor,
                                            int3 quaddims,
                                            int3 quadnum,
                                            bool[] iscorrupt,
                                            [MarshalAs(UnmanagedType.AnsiBStr)] string logpath);
    }
}
