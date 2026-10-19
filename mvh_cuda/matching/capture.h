#pragma once
#include <string>

namespace mvh_matching {
// Opt-in diagnostic. Production searches do no capture work by default.
void setCaptureEnabled(bool enabled);
void startCapture(const std::string& runDirectory);
}
