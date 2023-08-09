window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {

        function1: function (style) {
            // Get the root element
            var r = document.querySelector(':root');
            r.style.setProperty('--primary-color', style["background-color"])
            var darker = this.function2(style["background-color"], 0.5)
            r.style.setProperty('--primary-hover-color', darker)
            return "";
        },

        function2: function modifyRGB(inputRGB, multiplier) {
            const valuesStr = inputRGB.substring(inputRGB.indexOf("(") + 1, inputRGB.indexOf(")")).split(",");
            const values = [];
            for (let i = 0; i < valuesStr.length; i++) {
                values[i] = parseInt(valuesStr[i].trim());
                values[i] = Math.round(values[i] * multiplier);
            }

            return `rgb(${values[0]}, ${values[1]}, ${values[2]})`;
        }

    }

});

