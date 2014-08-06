using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Text;
using System.Threading;
using System.Xml;
using System.Windows.Threading;
using System.Globalization;
using System.Windows;
using System.Diagnostics;
using System.IO;
using Microsoft.Win32;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Effects;
using System.Windows.Data;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Sparta
{
    static class Helper
    {
        public static IFormatProvider NativeFormat = CultureInfo.InvariantCulture.NumberFormat;
        public static IFormatProvider NativeDateTimeFormat = CultureInfo.InvariantCulture.DateTimeFormat;
        public static float ParseFloat(string value)
        {
            return float.Parse(value, NativeFormat);
        }
        public static double ParseDouble(string value)
        {
            return double.Parse(value, NativeFormat);
        }
        public static int ParseInt(string value)
        {
            return int.Parse(value, NativeFormat);
        }
        public static Int64 ParseInt64(string value)
        {
            return Int64.Parse(value, NativeFormat);
        }
        public static decimal ParseDecimal(string value)
        {
            return decimal.Parse(value, NativeFormat);
        }
        public static DateTime ParseDateTime(string value)
        {
            return DateTime.Parse(value, NativeDateTimeFormat);
        }

        public static float ToRad = (float)Math.PI / 180.0f;
        public static float ToDeg = 180.0f / (float)Math.PI;

        public static void Swap<T>(ref T lhs, ref T rhs)
        {
            T temp;
            temp = lhs;
            lhs = rhs;
            rhs = temp;
        }
    }
}

namespace K2AlignDaemon
{ 
    [StructLayout(LayoutKind.Sequential)]
    public struct int3
    {
        public int X, Y, Z;

        public int3(int x, int y, int z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public ulong Elements()
        {
            return (ulong)X * (ulong)Y * (ulong)Z;
        }

        public uint ElementN(int3 position)
        {
            return ((uint)position.Z * (uint)Y + (uint)position.Y) * (uint)X + (uint)position.X;
        }

        public ulong ElementNLong(int3 position)
        {
            return ((ulong)position.Z * (ulong)Y + (ulong)position.Y) * (ulong)X + (ulong)position.X;
        }
    }
}