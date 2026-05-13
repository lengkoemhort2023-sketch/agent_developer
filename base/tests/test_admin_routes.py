from django.test import SimpleTestCase


class AdminRoutesTests(SimpleTestCase):
    def test_admin_without_trailing_slash_redirects(self):
        response = self.client.get("/admin", follow=False)

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response["Location"], "/admin/")
