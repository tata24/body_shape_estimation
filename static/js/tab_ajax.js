$(function () {
    $('#img_rectification').click(function () {
        img_rec_ajax = $.ajax({
            processData: false,
            contentType: false,
            url: '/img_rectification/',
            type: 'get',
            success: function (data) {
                data = JSON.parse(data);
                img1_left_base64 = 'data:image/jpg;base64,' + data['img1_left_base64'];
                img1_right_base64 = 'data:image/jpg;base64,' + data['img1_right_base64'];
                img2_left_base64 = 'data:image/jpg;base64,' + data['img2_left_base64'];
                img2_right_base64 = 'data:image/jpg;base64,' + data['img2_right_base64'];
                document.getElementById('tab2_pic1').style.visibility = 'visible';
                document.getElementById('tab2_pic1').src = img1_left_base64;
                document.getElementById('tab2_pic2').style.visibility = 'visible';
                document.getElementById('tab2_pic2').src = img1_right_base64;
                document.getElementById('tab2_pic3').style.visibility = 'visible';
                document.getElementById('tab2_pic3').src = img2_left_base64;
                document.getElementById('tab2_pic4').style.visibility = 'visible';
                document.getElementById('tab2_pic4').src = img2_right_base64;
            },
            error: function (data) {
                console.log('发生错误！');
            }
        })
        $.when(img_rec_ajax).done(function () {
            alert('矫正完成！');
        });
    });
});

$(function () {
    $('#person_seg').click(function () {
        var person_seg_ajax = new Array();
        for (var i = 0; i < 4; i++) {
            var post_data;
            if (i == 0) {
                post_data = {"id": i};
            } else if (i == 1) {
                post_data = {"id": i};
            } else if (i == 2) {
                post_data = {"id": i};
            } else if (i == 3) {
                post_data = {"id": i};
            }
            person_seg_ajax[i] = $.ajax({
                contentType: "application/json",
                url: '/img_seg/',
                type: 'post',
                headers: {'X-CSRFToken': $('input[name=csrfmiddlewaretoken]').val()},
                dataType: "json",
                data: JSON.stringify(post_data),
                success: function (data) {
                    data = JSON.parse(JSON.stringify(data));
                    img_base64 = 'data:image/jpg;base64,' + data['img_base64'];
                    id = Number(data['id']);
                    if (id == 0) {
                        document.getElementById('tab3_pic1').style.visibility = 'visible';
                        document.getElementById('tab3_pic1').src = img_base64;
                    } else if (id == 1) {
                        document.getElementById('tab3_pic2').style.visibility = 'visible';
                        document.getElementById('tab3_pic2').src = img_base64;
                    } else if (id == 2) {
                        document.getElementById('tab3_pic3').style.visibility = 'visible';
                        document.getElementById('tab3_pic3').src = img_base64;
                    } else if (id == 3) {
                        document.getElementById('tab3_pic4').style.visibility = 'visible';
                        document.getElementById('tab3_pic4').src = img_base64;
                    }
                },
                error: function (data) {
                    console.log('发生错误！');
                }
            })
        }
        $.when(person_seg_ajax[0], person_seg_ajax[1],
            person_seg_ajax[2], person_seg_ajax[3]).done(function () {
            alert('分割完成！');
        });
    });
});


$(function () {
    $('#person_key').click(function () {
        var person_key_ajax = new Array();
        for (var i = 0; i < 4; i++) {
            var post_data;
            if (i == 0) {
                post_data = {"id": i};
            } else if (i == 1) {
                post_data = {"id": i};
            } else if (i == 2) {
                post_data = {"id": i};
            } else if (i == 3) {
                post_data = {"id": i};
            }
            person_key_ajax[i] = $.ajax({
                contentType: "application/json",
                url: '/img_key/',
                type: 'post',
                headers: {'X-CSRFToken': $('input[name=csrfmiddlewaretoken]').val()},
                dataType: "json",
                data: JSON.stringify(post_data),
                success: function (data) {
                    data = JSON.parse(JSON.stringify(data));
                    img_base64 = 'data:image/jpg;base64,' + data['img_base64'];
                    id = Number(data['id']);
                    if (id == 0) {
                        document.getElementById('tab4_pic1').style.visibility = 'visible';
                        document.getElementById('tab4_pic1').src = img_base64;
                    } else if (id == 1) {
                        document.getElementById('tab4_pic2').style.visibility = 'visible';
                        document.getElementById('tab4_pic2').src = img_base64;
                    } else if (id == 2) {
                        document.getElementById('tab4_pic3').style.visibility = 'visible';
                        document.getElementById('tab4_pic3').src = img_base64;
                    } else if (id == 3) {
                        document.getElementById('tab4_pic4').style.visibility = 'visible';
                        document.getElementById('tab4_pic4').src = img_base64;
                    }
                },
                error: function (data) {
                    console.log('发生错误！');
                }
            })
        }
        $.when(person_key_ajax[0], person_key_ajax[1],
            person_key_ajax[2], person_key_ajax[3]).done(function () {
            alert('关键点检测完成！');
        });
    });
});


$(function () {
    $('#person_measure').click(function () {
        measure = $.ajax({
            contentType: "application/json",
            dataType: "json",
            url: '/measure/',
            type: 'get',
            headers: {'X-CSRFToken': $('input[name=csrfmiddlewaretoken]').val()},
            success: function (data) {
                data = JSON.parse(JSON.stringify(data));
                var img1_base64 = 'data:image/jpg;base64,' + data['img1_base64'];
                var img2_base64 = 'data:image/jpg;base64,' + data['img2_base64'];

                var h = (Number(data['height']) / 10).toFixed(2);
                var shoulder = (Number(data['shoulder']) / 10).toFixed(2);
                var d_left_arm = (Number(data['d_left_arm']) / 10).toFixed(2);
                var d_left_forearm = (Number(data['d_left_forearm']) / 10).toFixed(2);
                var d_right_arm = (Number(data['d_right_arm']) / 10).toFixed(2);
                var d_right_forearm = (Number(data['d_right_forearm']) / 10).toFixed(2);
                var waist_thickness = (Number(data['waist_thickness']) / 10).toFixed(2);
                var waist_width = (Number(data['waist_width']) / 10).toFixed(2);
                var waist = (Number(data['waist']) / 10).toFixed(2);
                var d = (Number(data['d']) / 10).toFixed(2);

                document.getElementById('height').innerText = h;
                document.getElementById('shoulder').innerText = shoulder;
                document.getElementById('d_left_arm').innerText = d_left_arm;
                document.getElementById('d_left_forearm').innerText = d_left_forearm;
                document.getElementById('d_right_arm').innerText = d_right_arm;
                document.getElementById('d_right_forearm').innerText = d_right_forearm;
                document.getElementById('waist_thickness').innerText = waist_thickness;
                document.getElementById('waist_width').innerText = waist_width;
                document.getElementById('d').innerText = d;
                document.getElementById('waist').innerText = waist;

                document.getElementById('tab5_pic1').style.visibility = 'visible';
                document.getElementById('tab5_pic1').src = img1_base64;
                document.getElementById('tab5_pic2').style.visibility = 'visible';
                document.getElementById('tab5_pic2').src = img2_base64;
            },
            error: function (data) {
                console.log('发生错误！');
            }
        });
        $.when(measure).done(function () {
            alert('测量完成！');
        });
    });
});



