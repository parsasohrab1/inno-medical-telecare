import unittest
import numpy as np
from inno_medical_telecare import core


class TestTelecare(unittest.TestCase):
    def test_critical_rules(self):
        self.assertTrue(core.critical(glucose=40))
        self.assertTrue(core.critical(glucose=350))
        self.assertTrue(core.critical(sbp=185, dbp=100))
        self.assertEqual(core.critical(glucose=110, sbp=125, dbp=80), [])

    def test_stable_patient_no_trend_alert(self):
        rng = np.random.default_rng(4)
        v = 130 + rng.normal(0, 10, 120)
        self.assertLessEqual(len(core.trend_alerts(v, np.zeros(120, bool))), 1)

    def test_drift_detected(self):
        rng = np.random.default_rng(5)
        v = 130 + rng.normal(0, 10, 120); v[60:] += np.clip(np.arange(60), 0, 15) * 4
        a = core.trend_alerts(v, np.zeros(120, bool))
        self.assertTrue(a and 60 <= a[0] <= 80)

    def test_missed_reading_reminder(self):
        m = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 1], bool)
        self.assertEqual(core.missed_reading_reminders(m), [3, 8])

    def test_routing(self):
        self.assertIn("physician", core.route("critical"))
        self.assertIn("behvarz", core.route("trend"))
