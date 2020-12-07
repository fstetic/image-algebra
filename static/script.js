
// https://codepen.io/mobifreaks/pen/LIbca
// function for displaying image on upload
function readURL(input) {
	if (input.files && input.files[0]) {
		let reader = new FileReader();
		reader.onload = function (e) {
			$('#img').attr('src', e.target.result);
			getResult(e.target.result)
		};
		reader.readAsDataURL(input.files[0]);
	}
}

// function for sending image to backend and updating label with equation and result
function getResult(image){
	let formData = new FormData()
	// https://stackoverflow.com/questions/45830815/is-there-a-built-in-function-to-extract-image-data-from-data-uri
	// removing data:[<media type>][;base64],
	let base64Image = image.replace(/^data:image\/(png|jpg|jpeg);base64,/, '')
	formData.append('img', base64Image)
	$.ajax({
		url: '/script',
		method: 'POST',
		contentType: false,
		processData: false,
		data: formData,
		}).then(function(response){
			// displaying equation and result
			document.getElementById('label').textContent = `${response['e']} = ${response['r']}`
			})
}

