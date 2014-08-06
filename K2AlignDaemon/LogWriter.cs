using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace K2AlignDaemon
{
    public class LogWriter : IDisposable
    {
        TextWriter Writer;

        public LogWriter(string path)
        {
            Writer = new StreamWriter(path, true);
        }

        public void Dispose()
        {
            if (Writer != null)
            {
                Writer.Close();
                Writer = null;
            }
        }

        public void Write(string message, LogMsgType type = LogMsgType.Success)
        {
            lock (Writer)
            {
                Writer.WriteLine(DateTime.Now.ToString("MM/dd/yy HH:mm:ss") + ", " + type.ToString() + ": " + message);
                Writer.Flush();
            }
        }

        public void Write(Exception e)
        {
            Write(e.Message + "\n" + e.Source + "\n" + e.StackTrace, LogMsgType.Error);
        }
    }

    public enum LogMsgType
    { 
        Success = 1 << 0,
        Warning = 1 << 1,
        Error = 1 << 2
    }
}
