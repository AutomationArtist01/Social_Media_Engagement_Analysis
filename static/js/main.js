/* ============================================================
   SocialPulse — client-side interactions
   ============================================================ */
(function () {
  "use strict";

  /**
   * Close the mobile nav menu after a link is tapped so the page isn't
   * left with the menu covering content.
   */
  function wireMobileNav() {
    var toggle = document.getElementById("nav-toggle");
    if (!toggle) return;
    document.querySelectorAll(".nav-links a").forEach(function (link) {
      link.addEventListener("click", function () {
        toggle.checked = false;
      });
    });
  }

  /**
   * Lightweight client-side validation + fake submit for the contact form.
   * No data is sent anywhere — this is a demonstration form.
   */
  function wireContactForm() {
    var form = document.getElementById("contact-form");
    if (!form) return;
    var status = document.getElementById("contact-status");

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var name = form.name.value.trim();
      var email = form.email.value.trim();
      var message = form.message.value.trim();

      var emailOk = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

      if (!name || !message || !emailOk) {
        status.textContent = "Please fill in all fields with a valid email.";
        status.className = "form-status err";
        return;
      }

      status.textContent =
        "Thanks, " + name + "! Your message has been received (demo).";
      status.className = "form-status ok";
      form.reset();
    });
  }

  /**
   * Disable the predict button while the form submits to prevent
   * double-submissions and give the user feedback.
   */
  function wirePredictForm() {
    var form = document.getElementById("predict-form");
    if (!form) return;
    form.addEventListener("submit", function () {
      var btn = form.querySelector('button[type="submit"]');
      if (btn) {
        btn.disabled = true;
        btn.textContent = "Predicting…";
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    wireMobileNav();
    wireContactForm();
    wirePredictForm();
  });
})();
