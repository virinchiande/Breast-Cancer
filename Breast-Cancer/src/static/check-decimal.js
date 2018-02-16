function CheckDecimal()
{
var isvalid = true
var radius_mean = document.form1.A1.value;
var perimeter_mean = document.form1.A2.value;
var concave_points_mean = document.form1.A3.value;
var radius_worst = document.form1.A4.value;
var texture_worst = document.form1.A5.value;
var perimeter_worst = document.form1.A6.value;
var area_worst = document.form1.A7.value;
var smoothness_worst = document.form1.A8.value;
var symmetry_worst = document.form1.A9.value;
var concave_points_worst = document.form1.A10.value;
console.log("inputtxt is:");
console.log("radius_mean"+perimeter_mean)
var decimal=  /^-?(0|[1-9]\d*)?(\.\d+)?(?<=\d)$/;
if(!radius_mean.match(decimal))
{
isvalid = false
console.log("true")
}
if(!perimeter_mean.match(decimal))
{
isvalid = false
console.log("true")
}
if(!concave_points_mean.match(decimal))
{
isvalid = false
console.log("true")
}
if(!radius_worst.match(decimal))
{
isvalid = false
console.log("true")
}
if(!texture_worst.match(decimal))
{
isvalid = false
console.log("true")
}
if(!perimeter_worst.match(decimal))
{
isvalid = false
console.log("true")
}
if(!area_worst.match(decimal))
{
isvalid = false
console.log("true")
}
if(!smoothness_worst.match(decimal))
{
isvalid = false
console.log("true")
}
if(!symmetry_worst.match(decimal))
{
isvalid = false
console.log("true")
}
if(!concave_points_worst.match(decimal))
{
isvalid = false
console.log("true")
}

if(!isvalid)
{
alert('Enter only Integers/Floats...!')

}
return isvalid
}






