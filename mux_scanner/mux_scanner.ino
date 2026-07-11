// PCA9548 I2C multiplexer channel scanner for ESP32
// Scans all 8 mux channels and reports every I2C device address found on each.
// Wiring: ESP32 GPIO21 -> SDA, GPIO22 -> SCL, both to PCA9548 (STEMMA QT).
// MLX90393 magnetometer connects to one PCA9548 downstream channel via STEMMA QT.

#include <Wire.h>

constexpr uint8_t MUX_ADDR = 0x70;
constexpr uint8_t SDA_PIN = 21;
constexpr uint8_t SCL_PIN = 22;
constexpr uint8_t MLX90393_ADDR = 0x0C; // default Adafruit MLX90393 address (A0/A1 low)

void muxSelectChannel(uint8_t channel) {
  if (channel > 7) return;
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

void muxDisableAll() {
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(0x00);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  delay(500);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  Serial.println();
  Serial.println(F("PCA9548 I2C Mux Scanner"));
  Serial.println(F("========================"));

  // Confirm the mux itself responds before touching channels.
  Wire.beginTransmission(MUX_ADDR);
  uint8_t muxErr = Wire.endTransmission();
  if (muxErr == 0) {
    Serial.print(F("Mux found at 0x"));
    Serial.println(MUX_ADDR, HEX);
  } else {
    Serial.print(F("WARNING: no response from mux at 0x"));
    Serial.print(MUX_ADDR, HEX);
    Serial.print(F(" (I2C error "));
    Serial.print(muxErr);
    Serial.println(F("). Check wiring/pull-ups before trusting the scan below."));
  }
  Serial.println();
}

void loop() {
  for (uint8_t channel = 0; channel < 8; channel++) {
    Serial.print(F("Channel "));
    Serial.print(channel);
    Serial.println(F(":"));

    muxSelectChannel(channel);
    delay(5); // let the mux settle

    uint8_t foundCount = 0;
    for (uint8_t addr = 1; addr < 127; addr++) {
      if (addr == MUX_ADDR) continue; // skip the mux's own address

      Wire.beginTransmission(addr);
      uint8_t err = Wire.endTransmission();

      if (err == 0) {
        foundCount++;
        Serial.print(F("  Found device at 0x"));
        if (addr < 16) Serial.print('0');
        Serial.print(addr, HEX);

        if (addr == MLX90393_ADDR) {
          Serial.print(F("  <-- likely MLX90393 magnetometer"));
        }
        Serial.println();
      }
    }

    if (foundCount == 0) {
      Serial.println(F("  (no devices found)"));
    }
  }

  muxDisableAll();

  Serial.println();
  Serial.println(F("Scan complete. Rescanning in 5 seconds..."));
  Serial.println();
  delay(5000);
}
