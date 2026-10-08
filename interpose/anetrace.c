// anetrace.c — dyld interposer for the ANE user-space path.
//
// Logs every IOKit user-client call, memory map, IOSurface create/lock and
// XPC send made by the process, and dumps the bytes of every buffer it can
// reach, so the traffic between ANEForge -> Espresso/AppleNeuralEngine -> the
// driver can be inventoried offline.
//
// Build:  clang -O1 -shared -fPI