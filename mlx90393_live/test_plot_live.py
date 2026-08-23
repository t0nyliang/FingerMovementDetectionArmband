import math
import unittest

from mlx90393_live.plot_live import SENSOR_COUNT, parse_packet


class ParsePacketTests(unittest.TestCase):
    def test_parses_four_sensor_frame(self):
        values = ",".join(str(index + 0.5) for index in range(12))
        packet = parse_packet(f"FRAME,7,140000,{values}")

        self.assertIsNotNone(packet)
        self.assertEqual(packet.source, "FRAME")
        self.assertEqual(packet.sequence, 7)
        self.assertEqual(packet.device_us, 140000)
        self.assertEqual(len(packet.readings), SENSOR_COUNT)
        self.assertEqual(packet.readings[0], (0.5, 1.5, 2.5))
        self.assertEqual(packet.readings[3], (9.5, 10.5, 11.5))

    def test_parses_nan_for_unavailable_sensor(self):
        packet = parse_packet(
            "FRAME,8,160000,1,2,3,nan,nan,nan,7,8,9,10,11,12"
        )

        self.assertIsNotNone(packet)
        self.assertTrue(all(math.isnan(value) for value in packet.readings[1]))
        self.assertEqual(packet.readings[2], (7.0, 8.0, 9.0))

    def test_expands_legacy_sample_to_sensor_zero(self):
        packet = parse_packet("SAMPLE,9,180000,1.25,2.5,3.75")

        self.assertIsNotNone(packet)
        self.assertEqual(packet.source, "SAMPLE")
        self.assertEqual(packet.readings[0], (1.25, 2.5, 3.75))
        for reading in packet.readings[1:]:
            self.assertTrue(all(math.isnan(value) for value in reading))

    def test_expands_legacy_data_to_sensor_zero(self):
        packet = parse_packet("DATA,-1,0,1")

        self.assertIsNotNone(packet)
        self.assertIsNone(packet.sequence)
        self.assertIsNone(packet.device_us)
        self.assertEqual(packet.readings[0], (-1.0, 0.0, 1.0))

    def test_parses_bno085_motion_packet(self):
        packet = parse_packet("MOTION,11,220000,1,2,3,4,5,6")

        self.assertIsNotNone(packet)
        self.assertEqual(packet.source, "MOTION")
        self.assertEqual(packet.sequence, 11)
        self.assertEqual(packet.device_us, 220000)
        self.assertEqual(packet.values, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    def test_rejects_status_malformed_and_partial_packets(self):
        self.assertIsNone(parse_packet("READY,protocol=FRAME_v1"))
        self.assertIsNone(parse_packet("FRAME,1,20,1,2,3"))
        self.assertIsNone(
            parse_packet("FRAME,1,20,1,2,3,4,5,6,7,8,9,10,11,bad")
        )
        self.assertIsNone(parse_packet("SAMPLE,1,20,1,2"))
        self.assertIsNone(parse_packet("MOTION,1,20,1,2,3"))


if __name__ == "__main__":
    unittest.main()
