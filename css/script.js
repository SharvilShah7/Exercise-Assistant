$(document).ready(function(){
    $('.category').each(function(index){
        $(this).delay(200 * index).animate({
            opacity: 1,
            translateY: 0
        }, 1000);
    });
});


// Initialization for ES Users
// import { Collapse, Ripple, initMDB } from "mdb-ui-kit";

// initMDB({ Collapse, Ripple });

window.onscroll = function() {scrollFunction()};

function scrollFunction() {
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
        document.getElementById("scrollToTopBtn").style.display = "block";
    } else {
        document.getElementById("scrollToTopBtn").style.display = "none";
    }
}

function scrollToTop() {
    document.body.scrollTop = 0;
    document.documentElement.scrollTop = 0;
}

$('form[name=login_form]').submit(function (e) {

    var $form = $(this);
    var $error = $form.find('.error');
    var data = $form.serialize();
    $.ajax({
        url: '/login_validation',
        type: 'POST',
        data: data,
        dataType: 'json',
        success: function (resp) {
            window.location.href = '/';
        },
        error: function (resp) {
            $error.text(resp.responseJSON.error).removeClass('error--hidden');
        }
    });
    e.preventDefault();

});

$('form[name=signup_form]').submit(function (e) {

    var $form = $(this);
    var $error = $form.find('.error');
    var data = $form.serialize();

    $.ajax({
        url: '/signup_validation',
        type: 'POST',
        data: data,
        dataType: 'json',
        success: function (resp) {
            console.log('success from signup');
            window.location.href = '/login/';
        },
        error: function (resp) {
            console.log('signup error')
            $error.text(resp.responseJSON.error).removeClass('error--hidden');
        }
    });
    e.preventDefault();

});