$(function () {
    $('#chooseImage0').on('change', function () {
        var name = "image_raw_1";
        upload(this, name);
        // 获取用户最后一次选择的图片
        var choose_file = $(this)[0].files[0];
        // 创建一个新的FileReader对象，用来读取文件信息
        var reader = new FileReader();
        // 读取用户上传的图片的路径
        reader.readAsDataURL(choose_file);
        // 读取完毕之后，将图片的src属性修改成用户上传的图片的本地路径
        reader.onload = function (e) {
            document.getElementById('upload1').style.visibility = 'visible';
            document.getElementById('upload1').src = e.target.result;
        }
    });

    $('#chooseImage1').on('change', function () {
        var name = "image_raw_2";
        upload(this, name);
        // 获取用户最后一次选择的图片
        var choose_file = $(this)[0].files[0];
        // 创建一个新的FileReader对象，用来读取文件信息
        var reader = new FileReader();
        // 读取用户上传的图片的路径
        reader.readAsDataURL(choose_file);
        // 读取完毕之后，将图片的src属性修改成用户上传的图片的本地路径
        reader.onload = function (e) {
            document.getElementById('upload2').style.visibility = 'visible';
            document.getElementById('upload2').src = e.target.result;
        }
    });

    function upload(input, name) {
        var formdata = new FormData();
        formdata.append("image", $(input)[0].files[0]);
        formdata.append("name", name);
        formdata.append("csrfmiddlewaretoken", $("[name='csrfmiddlewaretoken']").val());
        $.ajax({
            processData: false,
            contentType: false,
            url: '/uploadImg/',
            type: 'post',
            data: formdata,
            dataType: "json",
            success: function (arg) {
                alert('上传成功！')
            },
            error: function () {
                alert("访问繁忙，请重试")
            }
        })
    }
});

