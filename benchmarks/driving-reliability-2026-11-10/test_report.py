import copy
import unittest

from generate_report import zero_contact_mission


class MissionOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.summary = {'discipline': 'skidpad', 'result': 'finished', 'complete': True, 'pass': True,
                        'physical_cone_hits': 0, 'oc': 0, 'seconds_simulated': 50.3,
                        'route': {'accepted_route': ['right', 'right', 'left', 'left'], 'complete': True,
                                  'valid': True, 'exited': True, 'stopped_in_zone': True}}

    def test_legacy_pass_with_nine_contacts_is_rejected(self):
        self.summary['physical_cone_hits'] = 9
        self.assertFalse(zero_contact_mission(self.summary))

    def test_clean_full_mission_and_each_incomplete_stop(self):
        self.assertTrue(zero_contact_mission(self.summary))
        for key in ('complete', 'valid', 'exited', 'stopped_in_zone'):
            failed = copy.deepcopy(self.summary)
            failed['route'][key] = False
            self.assertFalse(zero_contact_mission(failed), key)

    def test_pilot_and_partial_rr_are_not_full_missions(self):
        self.summary['pilot_only'] = True
        self.assertFalse(zero_contact_mission(self.summary))
        self.summary.pop('pilot_only')
        self.summary['route']['accepted_route'] = ['right', 'right']
        self.assertFalse(zero_contact_mission(self.summary))

    def test_timeout_and_off_course_block_clean_track(self):
        summary = {'discipline': 'trackdrive', 'result': 'finished', 'complete': True, 'n_laps': 1,
                   'physical_cone_hits': 0, 'oc': 0, 'seconds_simulated': 56.25}
        self.assertTrue(zero_contact_mission(summary))
        for key, value in [('result', 'timeout'), ('oc', 1), ('seconds_simulated', 120.05), ('n_laps', 0)]:
            self.assertFalse(zero_contact_mission({**summary, key: value}), key)


if __name__ == '__main__':
    unittest.main()
