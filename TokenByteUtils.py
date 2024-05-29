class TokenByteUtils:
    BYTE_ENCODER = {
        33: "!", 34: "\"", 35: "#", 36: "$", 37: "%", 38: "&", 39: "'", 40: "(", 41: ")", 42: "*", 43: "+", 
        44: ",", 45: "-", 46: ".", 47: "/", 48: "0", 49: "1", 50: "2", 51: "3", 52: "4", 53: "5", 54: "6", 
        55: "7", 56: "8", 57: "9", 58: ":", 59: ";", 60: "<", 61: "=", 62: ">", 63: "?", 64: "@", 65: "A", 
        66: "B", 67: "C", 68: "D", 69: "E", 70: "F", 71: "G", 72: "H", 73: "I", 74: "J", 75: "K", 76: "L", 
        77: "M", 78: "N", 79: "O", 80: "P", 81: "Q", 82: "R", 83: "S", 84: "T", 85: "U", 86: "V", 87: "W", 
        88: "X", 89: "Y", 90: "Z", 91: "[", 92: "\\", 93: "]", 94: "^", 95: "_", 96: "`", 97: "a", 98: "b", 
        99: "c", 100: "d", 101: "e", 102: "f", 103: "g", 104: "h", 105: "i", 106: "j", 107: "k", 108: "l", 
        109: "m", 110: "n", 111: "o", 112: "p", 113: "q", 114: "r", 115: "s", 116: "t", 117: "u", 118: "v", 
        119: "w", 120: "x", 121: "y", 122: "z", 123: "{", 124: "|", 125: "}", 126: "~", 161: "\u00a1", 
        162: "\u00a2", 163: "\u00a3", 164: "\u00a4", 165: "\u00a5", 166: "\u00a6", 167: "\u00a7", 168: "\u00a8", 
        169: "\u00a9", 170: "\u00aa", 171: "\u00ab", 172: "\u00ac", 174: "\u00ae", 175: "\u00af", 176: "\u00b0", 
        177: "\u00b1", 178: "\u00b2", 179: "\u00b3", 180: "\u00b4", 181: "\u00b5", 182: "\u00b6", 183: "\u00b7", 
        184: "\u00b8", 185: "\u00b9", 186: "\u00ba", 187: "\u00bb", 188: "\u00bc", 189: "\u00bd", 190: "\u00be", 
        191: "\u00bf", 192: "\u00c0", 193: "\u00c1", 194: "\u00c2", 195: "\u00c3", 196: "\u00c4", 197: "\u00c5", 
        198: "\u00c6", 199: "\u00c7", 200: "\u00c8", 201: "\u00c9", 202: "\u00ca", 203: "\u00cb", 204: "\u00cc", 
        205: "\u00cd", 206: "\u00ce", 207: "\u00cf", 208: "\u00d0", 209: "\u00d1", 210: "\u00d2", 211: "\u00d3", 
        212: "\u00d4", 213: "\u00d5", 214: "\u00d6", 215: "\u00d7", 216: "\u00d8", 217: "\u00d9", 218: "\u00da", 
        219: "\u00db", 220: "\u00dc", 221: "\u00dd", 222: "\u00de", 223: "\u00df", 224: "\u00e0", 225: "\u00e1", 
        226: "\u00e2", 227: "\u00e3", 228: "\u00e4", 229: "\u00e5", 230: "\u00e6", 231: "\u00e7", 232: "\u00e8", 
        233: "\u00e9", 234: "\u00ea", 235: "\u00eb", 236: "\u00ec", 237: "\u00ed", 238: "\u00ee", 239: "\u00ef", 
        240: "\u00f0", 241: "\u00f1", 242: "\u00f2", 243: "\u00f3", 244: "\u00f4", 245: "\u00f5", 246: "\u00f6", 
        247: "\u00f7", 248: "\u00f8", 249: "\u00f9", 250: "\u00fa", 251: "\u00fb", 252: "\u00fc", 253: "\u00fd", 
        254: "\u00fe", 255: "\u00ff", 0: "\u0100", 1: "\u0101", 2: "\u0102", 3: "\u0103", 4: "\u0104", 5: "\u0105", 
        6: "\u0106", 7: "\u0107", 8: "\u0108", 9: "\u0109", 10: "\u010a", 11: "\u010b", 12: "\u010c", 13: "\u010d", 
        14: "\u010e", 15: "\u010f", 16: "\u0110", 17: "\u0111", 18: "\u0112", 19: "\u0113", 20: "\u0114", 21: "\u0115", 
        22: "\u0116", 23: "\u0117", 24: "\u0118", 25: "\u0119", 26: "\u011a", 27: "\u011b", 28: "\u011c", 29: "\u011d", 
        30: "\u011e", 31: "\u011f", 32: "\u0120", 127: "\u0121", 128: "\u0122", 129: "\u0123", 130: "\u0124", 
        131: "\u0125", 132: "\u0126", 133: "\u0127", 134: "\u0128", 135: "\u0129", 136: "\u012a", 137: "\u012b", 
        138: "\u012c", 139: "\u012d", 140: "\u012e", 141: "\u012f", 142: "\u0130", 143: "\u0131", 144: "\u0132", 
        145: "\u0133", 146: "\u0134", 147: "\u0135", 148: "\u0136", 149: "\u0137", 150: "\u0138", 151: "\u0139", 
        152: "\u013a", 153: "\u013b", 154: "\u013c", 155: "\u013d", 156: "\u013e", 157: "\u013f", 158: "\u0140", 
        159: "\u0141", 160: "\u0142", 173: "\u0143"
    }

    BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}
