var isDOMLoaded = false
document.addEventListener("DOMContentLoaded", function() {
    // DOMContentLoaded event has fired
    isDOMLoaded = true;
    console.log("DOM content loaded.");
});


async function isPageLoaded() {
    return new Promise((resolve, reject) => {
        if (isDOMLoaded) {
            resolve();
        } else {
            setTimeout(isPageLoaded, 100)
        }


    });
}


let nightStyleElement = document.createElement('style');
document.head.appendChild(nightStyleElement);

async function clickSimulator(trace) {
    return new Promise((resolve, reject) => {
        var cursor = document.createElement('i');
        cursor.classList.add("fa-solid", "fa-hand-pointer", "fa", "fa-lg")
        cursor.id = "tut-cursor"
        var start = trace[0]
        var rect = start.getBoundingClientRect()
        var initialX = rect.left + window.scrollX;
        var initialY = rect.top + window.scrollY;
        cursor.style.position = "absolute";
        cursor.style.left = initialX + "px";
        cursor.style.top = initialY + "px";
        document.body.appendChild(cursor);
        var target = trace[1];
        console.log("running clickSimulator target: ", target)

        var targetRect = target.getBoundingClientRect()
        var x = targetRect.left + targetRect.width / 2 + window.scrollX;
        var y = targetRect.top + targetRect.height / 2 + window.scrollY;
        var moveX = x - initialX;
        var moveY = y - initialY;

        cursor.style.transform = `translate(${moveX}px, ${moveY}px)`;
        target.scrollIntoView({ behavior: "smooth", block: "center" });

        setTimeout(() => {
            var event = new MouseEvent('click', {
                bubbles: true,
                cancelable: true,
                view: window,
            });
            target.dispatchEvent(event);
            resolve(); // Resolve the Promise when click simulation is complete
        }, 1000);

        setTimeout(() => {
            document.body.removeChild(cursor);
        }, 1100);
    });
}

function simulateClickClass(className, nr, visibleId = null) {
    return  function () {
        return new Promise((resolve, reject) => {

            var attempts = 0; // Initialize attempts counter
            var clickFn2 = async function () {

                var nextbtn = document.getElementById("tut-next");
                console.log(nextbtn, "next-btn")
                var targets = document.getElementsByClassName(className);
                console.log(targets, "target")

                if (!nextbtn || targets.length <= nr) {
                    attempts++; // Increment attempts
                    setTimeout(clickFn2, 500); // Retry after a delay
                } else if (attempts < 5) {
                    var target = targets[nr]

                    await clickSimulator([nextbtn, target])
                    resolve();



                } else {
                    console.error("Element with id '" + id + "' not found after 5 attempts.");
                }
            }
            if (visibleId) {
                var vid = document.getElementsByClassName(visibleId);
                console.log(vid, "visibleID")
                if (vid.length > 0) {
                    resolve();

                } else {
                    clickFn2()
                }

            } else {
                clickFn2()
            }

        });
    }


}

function simulateClickSorting(columnName, mode = "down") {
    return function () {
        return new Promise((resolve, reject) => {
            var attempts = 0; // Initialize attempts counter
            var clickFn2 = async function () {

                var nextbtn = document.getElementById("tut-next");
                console.log(nextbtn, "next-btn")
                const thElement = document.querySelector('th[data-dash-column="' + columnName + '"]');
                const div = thElement.querySelector('.column-actions');
                const target = div.querySelector('.column-header--sort');
                console.log(target, "target Sorting")

                if (!nextbtn || !target) {
                    console.log("try again")
                    attempts++; // Increment attempts
                    setTimeout(clickFn2, 100); // Retry after a delay
                } else if (attempts < 5) {
                    const svg = target.querySelector('svg');
                    console.log(svg.classList, "svgClasses")

                    let sorted = svg.classList.contains(`fa-sort-${mode}`)
                    while (!sorted) {
                        await clickSimulator([nextbtn, target])
                        sorted = svg.classList.contains(`fa-sort-${mode}`)
                    }
                    resolve();
                } else {
                    console.error("Element with id '" + id + "' not found after 5 attempts.");
                    resolve();
                }
            }

            clickFn2()

        });
    }


}

function simulateClickTableInput(nr) {
    return  function () {
        return new Promise((resolve, reject) => {

            var attempts = 0; // Initialize attempts counter
            var clickFn2 = async function () {

                var nextbtn = document.getElementById("tut-next");
                console.log(nextbtn, "next-btn")
                var targets = document.querySelectorAll('tr');
                console.log(targets, "targets TableInput")

                if (!nextbtn || !targets || targets.length < nr || targets.length === 3) {
                    console.log("try again")
                    attempts++; // Increment attempts
                    setTimeout(clickFn2, 100); // Retry after a delay
                } else if (attempts < 5) {
                    var target = targets[nr].querySelector("input");
                    console.log(target, "targetofTableInput")
                    console.log(target.checked, "targetofTableInput is checked")
                    if (target.checked) {
                        resolve();
                    } else {
                        await clickSimulator([nextbtn, target])
                        resolve();

                    }



                } else {
                    console.error("Element with id '" + id + "' not found after 5 attempts.");
                    resolve();
                }
            }

            clickFn2()

        });
    }


}

function simulateClickFormCheckInput(nr) {
    return  function () {
        return new Promise((resolve, reject) => {

            var attempts = 0; // Initialize attempts counter
            var clickFn2 = async function () {

                var nextbtn = document.getElementById("tut-next");
                console.log(nextbtn, "next-btn")
                var targets = document.getElementsByClassName('form-check-label');
                console.log(targets, "targetIP")

                if (!nextbtn || !targets || targets.length < nr) {
                    console.log("???")
                    attempts++; // Increment attempts
                    setTimeout(clickFn2, 500); // Retry after a delay
                } else if (attempts < 5) {
                    console.log("foo")
                    var target = targets[nr];
                    if (target) {
                        await clickSimulator([nextbtn, target])
                        resolve();

                    } else {
                        attempts++;
                        setTimeout(clickFn2, 500); // Retry after a delay
                    }
                    console.log(target, "target")




                } else {
                    console.error("Element with id '" + id + "' not found after 5 attempts.");
                    resolve();
                }
            }

            clickFn2()

        });
    }


}

function simulateClickID(id, visibleId = null) {
    return function () {
        return new Promise((resolve, reject) => {
            var attempts = 0; // Initialize attempts counter
            var clickFn = async function () {
                var nextbtn = document.getElementById("tut-next");
                console.log(nextbtn, "next-btn")
                var target = document.getElementById(id);
                console.log(target, "target")

                if (!nextbtn || !target) {
                    attempts++; // Increment attempts
                    if (attempts < 5) {
                        setTimeout(clickFn, 100); // Retry after a delay
                    } else {
                        console.error("Element with id '" + id + "' not found after 5 attempts.");
                        reject(new Error("Element not found"));
                    }
                } else {
                    await clickSimulator([nextbtn, target])
                    console.log("RESOLVED")
                    resolve(); // Resolve the Promise when click simulation is successful
                }
            };

            if (visibleId) {
                var vid = document.getElementById(visibleId);
                console.log(vid, "visibleID")
                if (!vid) {
                    clickFn();
                } else {
                    resolve(); // Resolve the Promise immediately if visibleId is present
                }
            } else {
                clickFn();
            }
        });
    };
}


function multiClickInOne(fctArray) {
    return async function () {
        for (let i = 0; i < fctArray.length; i++) {
            await fctArray[i]();
        }
    };
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {

        restyleRadio: function(url) {
            var formCheckElements = document.getElementsByClassName("form-check");
            if (formCheckElements.length > 0) {
                for (var i = 0; i < formCheckElements.length; i++) {
                    formCheckElements[i].classList.add("col-4");
                    formCheckElements[i].classList.add("col-md-3");
                }
            }
            return ""


        },

        moveBtn: function (tbl, tab) {
            if (tab === "tab-2") {
                return ""
            } else {
                var btn = document.getElementById("reset-rows-btn")
                var container = document.getElementsByClassName("previous-next-container")[0];
                container.insertBefore(btn, container.firstChild);
                return ""
            }

        },

        styleSelectedTableRow: function (proteinKey, tbl_data) {
            const key = proteinKey.split("Protein ")[1];
            try {
                var tables = document.getElementById("tbl").querySelectorAll("table");

            } catch (error) {
                return ""
            }
            var table = tables[[tables.length - 1]];
            var rows = table.getElementsByTagName("tr");
            for (let i = 0; i < rows.length; i++) {
                var cells = rows[i].getElementsByTagName("td");
                rows[i].classList.remove('selected-row');

                // Check if the first column value (cells[0]) matches the custom key
                if (cells.length > 0 && cells[1].children[0].textContent === key) {
                    // You've found a match, you can do something with it
                    rows[i].classList.add("selected-row")
                }
            }
            return ""


        },
        styleTutorial: function (style2, style, fill_starts, fill2_starts) {
            const svgImage = document.getElementById('tutorial-rapdor-svg');
            style = this.rgbToHex(style)
            style2 = this.rgbToHex(style2)
            const fill = "fill:" + style;
            const fill2 = "fill:" + style2;
            if (svgImage) {

                const base64EncodedSvg = svgImage.getAttribute('src').replace(/^data:image\/svg\+xml;base64,/, '');
                const decodedSvg = atob(base64EncodedSvg);
                var modifiedSvg = decodedSvg;

                // Iterate over each index in fill_starts and apply substrReplace
                fill_starts.forEach(fill_start => {
                    modifiedSvg = this.substrReplace(modifiedSvg, fill_start, fill);
                });
                fill2_starts.forEach(fill_start => {
                    modifiedSvg = this.substrReplace(modifiedSvg, fill_start, fill2);
                });
                // if (!on) {
                //     modifiedSvg = this.substrReplace(modifiedSvg, black_start, "fill:#000000");
                //
                // } else {
                //     modifiedSvg = this.substrReplace(modifiedSvg, black_start, "fill:#f2f2f2");
                //
                // }
                svgImage.setAttribute('src', 'data:image/svg+xml;base64,' + btoa(modifiedSvg));
            }
            return ""
        },

        styleFlamingo: function (style2, style, fill_starts, fill2_starts) {
            const svgImage = document.getElementById('flamingo-svg');
            style = this.rgbToHex(style)
            style2 = this.rgbToHex(style2)
            const fill = "fill:" + style;
            const fill2 = "fill:" + style2;
            if (svgImage) {

                const base64EncodedSvg = svgImage.getAttribute('src').replace(/^data:image\/svg\+xml;base64,/, '');
                const decodedSvg = atob(base64EncodedSvg);
                var modifiedSvg = decodedSvg;

                // Iterate over each index in fill_starts and apply substrReplace
                fill_starts.forEach(fill_start => {
                    modifiedSvg = this.substrReplace(modifiedSvg, fill_start, fill);
                });
                fill2_starts.forEach(fill_start => {
                    modifiedSvg = this.substrReplace(modifiedSvg, fill_start, fill2);
                });
                // if (!on) {
                //     modifiedSvg = this.substrReplace(modifiedSvg, black_start, "fill:#000000");
                //
                // } else {
                //     modifiedSvg = this.substrReplace(modifiedSvg, black_start, "fill:#f2f2f2");
                //
                // }
                svgImage.setAttribute('src', 'data:image/svg+xml;base64,' + btoa(modifiedSvg));
            }
            return ""
        },

        rgbToHex: function rgbToHex(rgbString) {
            const match = rgbString.match(/^rgb\((\d+),\s*(\d+),\s*(\d+)\)$/);

            if (!match) {

                throw new Error('Invalid RGB string format');
            }

            const [, red, green, blue] = match;

            const hexRed = parseInt(red).toString(16).padStart(2, '0');
            const hexGreen = parseInt(green).toString(16).padStart(2, '0');
            const hexBlue = parseInt(blue).toString(16).padStart(2, '0');

            return `#${hexRed}${hexGreen}${hexBlue}`;
        },

        substrReplace: function replaceInRange(inputString, startCoordinate, replacement) {
            const startIndex = startCoordinate;
            const endIndex = startIndex + replacement.length;

            const newString = inputString.slice(0, startIndex) + replacement + inputString.slice(endIndex);

            return newString;
        },

        displayToolTip: function displayEllipsies(input_trigger) {
            var elements = document.querySelectorAll('.column-header-name');
            console.log(elements)
            elements.forEach(function (element) {
                    element.addEventListener('mouseover', function (event) {
                        if (element.scrollWidth > element.clientWidth) {
                            console.log(element.scrollWidth)
                            console.log(element.clientWidth)

                            var fullText = element.textContent;
                            var tooltip = document.createElement('div');
                            tooltip.textContent = fullText;
                            tooltip.classList.add('rtooltip');
                            tooltip.classList.add('databox');
                            document.body.appendChild(tooltip);

                            var x = event.pageX + 10; // Add 10px offset to avoid covering the mouse pointer
                            var y = event.pageY + 10;
                            tooltip.style.top = y + 'px';
                            tooltip.style.left = x + 'px';
                            tooltip.style.display = "block";
                        }
                    });

                    element.addEventListener('mouseout', function (event) {
                        var tooltip = document.querySelector('.rtooltip');
                        if (tooltip) {
                            tooltip.remove();
                        }
                    });
                })
            var elements = document.querySelectorAll('.dash-cell');
            elements.forEach(function (element) {
                    element.addEventListener('mouseover', function (event) {
                        if (element.scrollWidth > element.clientWidth) {

                            var fullText = element.childNodes[0].textContent;
                            var tooltip = document.createElement('div');
                            tooltip.textContent = fullText;
                            tooltip.classList.add('rtooltip');
                            tooltip.classList.add('databox');
                            document.body.appendChild(tooltip);

                            var x = event.pageX + 10; // Add 10px offset to avoid covering the mouse pointer
                            var y = event.pageY + 10;
                            tooltip.style.top = y + 'px';
                            tooltip.style.left = x + 'px';
                            tooltip.style.display = "block";
                        }
                    });

                    element.addEventListener('mouseout', function (event) {
                        var tooltip = document.querySelector('.rtooltip');
                        if (tooltip) {
                            tooltip.remove();
                        }
                    });
                })

            return ""
        },


        function2: function modifyRGB(inputRGB, multiplier) {
            const valuesStr = inputRGB.substring(inputRGB.indexOf("(") + 1, inputRGB.indexOf(")")).split(",");
            const values = [];
            for (let i = 0; i < valuesStr.length; i++) {
                values[i] = parseInt(valuesStr[i].trim());
                values[i] = Math.round(values[i] * multiplier);
            }

            return `rgb(${values[0]}, ${values[1]}, ${values[2]})`;
        },
        makebrighter: function makeRGBBrighter(inputRGB, percentage) {
            const valuesStr = inputRGB.substring(inputRGB.indexOf("(") + 1, inputRGB.indexOf(")")).split(",");
            const values = [];

            for (let i = 0; i < valuesStr.length; i++) {
                values[i] = parseInt(valuesStr[i].trim());
            }

            const diffR = 255 - values[0];
            const diffG = 255 - values[1];
            const diffB = 255 - values[2];

            const brighterR = Math.round(diffR * (percentage / 100));
            const brighterG = Math.round(diffG * (percentage / 100));
            const brighterB = Math.round(diffB * (percentage / 100));

            const newR = values[0] + brighterR;
            const newG = values[1] + brighterG;
            const newB = values[2] + brighterB;

            return `rgb(${newR}, ${newG}, ${newB})`;
        },

        nightMode: function changeMode(on, primaryColor, secondaryColor) {
            var r = document.querySelector(':root');
            r.style.setProperty('--primary-color', primaryColor)
            r.style.setProperty('--secondary-color', secondaryColor)
            let btn = document.getElementById("night-mode").querySelector('button')
            btn.classList.add("fa", "fa-solid", "fa")
            let styleSheet = nightStyleElement.sheet;
            let cssRule;
            while (styleSheet.cssRules.length) {
                styleSheet.deleteRule(0);
            }

            if (on) {
                r.style.setProperty('--r-text-color', "white")
                r.style.setProperty('--databox-color', "#181818")
                r.style.setProperty('--table-light', "#3a363d")
                r.style.setProperty('--table-dark', "#222023")
                r.style.setProperty('--button-color', "#222023")
                r.style.setProperty('--input-background-color', "#2f2f2f")
                r.style.setProperty('--background-color', "#3a3a3a")
                r.style.setProperty('--disabled-input', "#181818")
                var darker = this.function2(primaryColor, 0.5);
                var darker2 = this.function2(secondaryColor, 0.5);
                r.style.setProperty('--primary-hover-color', darker);
                r.style.setProperty('--secondary-hover-color', darker2);
                var table_head = this.function2(primaryColor, 0.05);
                r.style.setProperty('--table-head-color', table_head);
                btn.classList.add("fa-moon")
                btn.classList.remove("fa-sun")
                btn.style.backgroundColor = "black"
                btn.style.borderColor = "black"
                btn.style.color = "white"
                cssRule = '.d-light { display: none !important; }';
                styleSheet.insertRule(cssRule, 0);


            } else {
                r.style.setProperty('--r-text-color', "black")
                r.style.setProperty('--databox-color', "#fffdfd")
                r.style.setProperty('--table-light', "#e1e1e1")
                r.style.setProperty('--table-dark', "#c0c0c0")
                r.style.setProperty('--button-color', "#8f8f8f")
                r.style.setProperty('--input-background-color', "#e0e0e0")
                r.style.setProperty('--background-color', "#ffffff")
                var lighter = this.makebrighter(primaryColor, 50);
                var lighter2 = this.makebrighter(secondaryColor, 50);
                r.style.setProperty('--table-head-color', "#181818");
                r.style.setProperty('--primary-hover-color', lighter);
                r.style.setProperty('--secondary-hover-color', lighter2);
                r.style.setProperty('--disabled-input', "#a6a6a6")
                btn.classList.remove("fa-moon")
                btn.classList.add("fa-sun")
                btn.style.backgroundColor = "white"
                btn.style.borderColor = "white"
                btn.style.color = "var(--table-head-color)"
                cssRule = '.d-night { display: none !important; }';
                styleSheet.insertRule(cssRule, 0);






            }
            return ""
        },


        stepsDataCache: null,
        tutorialFile: null,

        textForStep: function textForStep(stepNumber) {
            return this.stepsDataCache[stepNumber]
        },


        toggleTutOverlay: function (){
            const overlay = document.getElementById('tut-overlay');
            const tutRow = document.getElementById('tut-row');
            tutRow.classList.toggle('d-none');
            overlay.classList.toggle('d-none');
            overlay.classList.toggle('shadow');
            this.resizeTutorial()
        },
        activateDisplayTutorial: function(btn, skip_btn, url, tutorial_dialog) {
          this.tutorialFile = "assets/tutorialDisplayMode.json";
          this.tutorialSteps = this.displayTutSteps;
          this.tutStartUp(tutorial_dialog)

        },

        tutStartUp: function(dialog) {
            var tutFlag = sessionStorage.getItem("tutorial-flag");
            if (!this.stepsDataCache) {
                this.stepsDataCache = dialog
            }
            if (dash_clientside.callback_context.triggered[0].prop_id === "url.pathname") {
                if (tutFlag === null || tutFlag === undefined) {
                    return ""

                }
            }
            this.toggleTutOverlay()

            if (dash_clientside.callback_context.triggered[0].prop_id === "tut-end.n_clicks") {
                sessionStorage.removeItem("tutorial-flag");
                this.removeHighlights()
                this.resizeTutorial()

            } else {
                sessionStorage.setItem("tutorial-flag", 1);
                isPageLoaded().then(() => {
                    this.loadTutorialStep(0)
                })
            }
        },

        activateTutorial: function (btn, skip_btn, url, tutorial_dialog) {
            this.tutorialFile = 'assets/tutorial.json';
            this.tutorialSteps = this.tutSteps;
            this.tutStartUp(tutorial_dialog)


            return ""

        },

        highlightDiv: function highlightDiv(highlightIDs, selectable, attempts = 0) {
            if (highlightIDs && attempts < 5) {
                highlightIDs.forEach(function (highlightID) {
                    var highlight = document.getElementById(highlightID);
                    if (highlight) {
                        if (highlightID === highlightIDs[0]) {
                            highlight.classList.add('highlighted');
                            highlight.scrollIntoView({ behavior: "smooth", block: "center" });
                            console.log(highlight, "highlighting")
                            if (selectable) {
                            highlight.classList.add('tut-selectable');
                        }

                        } else {
                            highlight.classList.add('highlighted-no-shadow');
                            console.log(highlight, "highlighting-noshadow")
                            if (selectable) {
                            highlight.classList.add('tut-selectable-no-shadow');
                        }

                        }


                    } else {
                        setTimeout(function () {
                            highlightDiv(highlightIDs, selectable, attempts + 1); // Call itself with the same array and increment attempts
                        }, 500);
                    }
                });
            } else {
                if (highlightIDs) {
                    console.log("Div not found after 5 attempts")
                }
            }
        },


        removeHighlights: function () {
            var highlightedElements = document.querySelectorAll('.highlighted, .highlighted-no-shadow');
            highlightedElements.forEach(function (element) {
                element.classList.remove('highlighted');
                element.classList.remove('tut-selectable');
                element.classList.remove('highlighted-no-shadow');
                element.classList.remove('tut-selectable-no-shadow');
            });

        },

        loadTutorialStep: function loadStep (step) {
            var ts = sessionStorage.getItem("tutorial-step");

            if (ts === null || ts === undefined) {
                // Set ts to zero
                ts = 0;
            }

            var overlay = document.getElementById("tut-overlay");
            overlay.classList.add('shadow');
            this.removeHighlights()

            ts = parseInt(ts);
            ts = ts + step;


            if (this.tutorialSteps.length > ts) {
                var [stepName, highlightID, page, selectable, runFunction, reverse] = this.tutorialSteps[ts];
                if (page) {
                    if (window.location.pathname !== page) {
                        sessionStorage.setItem("tutorial-step", ts);

                        window.location.href = page
                        return ""
                    }


                }
                console.log("runFunction", runFunction)
                if (runFunction && highlightID){
                    console.log("runFct", typeof runFunction)
                    console.log("bla")
                    runFunction().then(() => {
                        overlay.classList.remove('shadow');

                        this.highlightDiv(highlightID, selectable);
                    })
                } else if (runFunction) {
                    runFunction()
                } else {
                    this.highlightDiv(highlightID, selectable);
                    overlay.classList.remove('shadow');


                }
                var text = document.getElementById("tut-text");
                var textFS = this.textForStep(ts)


                text.innerHTML = textFS
                var header = document.getElementById("tut-head");
                header.textContent = stepName;
                var trow = document.getElementById("tut-row");
                var step = document.getElementById("tut-step");
                var svgImage = document.getElementById('tutorial-rapdor-svg');
                var rapdorDiv = document.getElementById("TutorialRapdor")
                step.textContent = `${ts+1}/${this.tutorialSteps.length}`

                if (reverse) {
                    trow.classList.add("flex-row-reverse")
                    svgImage.style.transform = 'scale(-1, 1)';
                    rapdorDiv.classList.add("justify-content-end");
                } else {
                    trow.classList.remove("flex-row-reverse")
                    rapdorDiv.classList.remove("justify-content-end");

                    svgImage.style.transform = "none"

                }

            } else {
                // Key ts does not exist in this.tutorialSteps
                ts = 0;
                sessionStorage.setItem("tutorial-step", ts);
                sessionStorage.removeItem("tutorial-flag");
                this.toggleTutOverlay();
                this.resizeTutorial()

                return ts

            }
            sessionStorage.setItem("tutorial-step", ts);


            return ts


        },

        tutorialStep: function (next, previous) {
            var ts
            if (dash_clientside.callback_context.triggered[0].prop_id === "tut-next.n_clicks") {
                ts = this.loadTutorialStep(1)
            } else {
                ts = this.loadTutorialStep(-1)

            }
            if (ts == 5) {
                return ts
            } else {
                return dash_clientside.no_update
            }
        },

        resizeTutorial: function () {
            const tutRow = document.getElementById('tut-row');
            const otherDiv = document.getElementById('footer-row');
            console.log(tutRow, "tutRow")

            // Function to update height of otherDiv
            function updateOtherDivHeight() {
                console.log(tutRow.clientHeight)
                otherDiv.style.minHeight = tutRow.clientHeight + 'px';
            }

            // Initial update
            updateOtherDivHeight();

            // Resize event listener for changes in screen size
            window.addEventListener('resize', updateOtherDivHeight);

            // MutationObserver to detect changes in tutRow size
            const observer = new MutationObserver(updateOtherDivHeight);
            observer.observe(tutRow, {attributes: true, childList: true, subtree: true});
        },

        tutorialSteps: null,
        displayTutSteps: [
            ["Tutorial", ["tut-overlay"], null, false, null],
            ["Tutorial", ["tut-overlay"], null, false, null],
            ["Analysis", null, "/analysis", false, null],
            ["Distribution", ["distribution-panel"], "/analysis", false, null],
            ["Distribution", ["table-tab", "table-tut", "distribution-panel"], "/analysis", false, simulateClickClass("dash-cell column-0", 1)],
            ["Distribution", ["rapdor-id"], "/analysis", false, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickClass("dash-cell column-0", 1, "selected-row")])],
            ["Distribution", ["distribution-panel"], "/analysis", true, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickClass("dash-cell column-0", 1, "selected-row")])],
            ["Distribution", ["distribution-panel"], "/analysis", true, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickClass("dash-cell column-0", 1, "selected-row")])],
            ["Distribution", ["distribution-panel"], "/analysis", true, null],
            ["Table", ["table-tut", "table-tab"], "/analysis", true, simulateClickID("table-tab", "table-tut")],
            ["Table", ["distribution-panel", "table-tut", "table-tab"], "/analysis", true, simulateClickID("table-tab", "table-tut")],
            ["Settings", ["selector-box-tut"], "/analysis", false, null, true],
            ["Settings", ["selector-box-tut"], "/analysis", false, null, true],
            ["Settings", ["kernel-tut"], "/analysis", false, null, true],
            ["Settings", ["distance-method-tut"], "/analysis", false, null, true],
            ["Settings", ["heatmap-box-tut"], "/analysis", false, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickClass("dash-cell column-0", 1, "selected-row")])],
            ["Settings", ["table-tut", "table-tab", "selector-box-tut"], "/analysis", false, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickSorting("Rank", "up")])],
            ["Settings", ["heatmap-box-tut"], "/analysis", false, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickSorting("Rank", "up"), simulateClickClass("dash-cell column-0", 1)])],
            ["Settings", ["distribution-panel"], "/analysis", false, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickSorting("Rank", "up"), simulateClickClass("dash-cell column-0", 1)])],
            ["Settings", ["export-tut"], "/analysis", true, null, true],
            ["Figure Creation", ["color-tut"], "/analysis", true, null, true],
            ["Figure Creation", ["table-tut", "table-tab"], "/analysis", true, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickTableInput( 3), simulateClickTableInput( 5)])],
            ["Figure Creation", null, "/figure_factory", false, simulateClickFormCheckInput(0)],
            ["Figure Creation", ["ff-tut-preview"], "/figure_factory", false, simulateClickID("ff-default", 1)],
            ["Figure Creation", ["ff-tut-preview"], "/figure_factory", false, multiClickInOne([simulateClickFormCheckInput(2), simulateClickID("ff-default", null) ])],
            ["Bubble Plot", null, "/analysis", true],
            ["Bubble Plot", ["distribution-panel", "dim-red-tab", "dim-red-tut"], "/analysis", true, simulateClickID("dim-red-tab", "dim-red-tut")],
            ["Bubble Plot", ["dim-red-tut", "distribution-panel", "dim-red-tab", ], "/analysis", true, simulateClickID("dim-red-tab", "dim-red-tut")],
            ["Bubble Plot", ["dim-red-tut", "distribution-panel", "dim-red-tab", ], "/analysis", true, simulateClickID("dim-red-tab", "dim-red-tut")],
            ["Finish",null, null, false, simulateClickID("dim-red-tab", "dim-red-tut")],




        ],
        // Steps are organized like this:
        // [step_name, ids_to_highlight, "page to go to", "can you click on highlighted divs", function to run]
        tutSteps: [
            ["Tutorial",["tut-overlay"], null, false, null],
            ["Tutorial", ["tut-overlay"], null, false, null],
            ["Tutorial", ["tut-overlay"], null, false, null],
            ["Data Upload", null, "/", false, null],
            ["Data Upload", ["from-csv", "from-csv-tab"], "/", false, simulateClickID("from-csv-tab",  "from-csv")],
            ["Data Upload", ["from-json", "from-json-tab"], "/", false, simulateClickID("from-json-tab", "from-json")],
            ["Data Upload", ["from-json", "from-json-tab"], "/", false, simulateClickID("from-json-tab", "from-json")],
            ["Analysis", null, "/analysis", false, null],
            ["Distribution", ["distribution-panel"], "/analysis", false, null],
            ["Distribution", ["table-tab", "table-tut", "distribution-panel"], "/analysis", false, simulateClickClass("dash-cell column-0", 1)],
            ["Distribution", ["rapdor-id"], "/analysis", false, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickClass("dash-cell column-0", 1, "selected-row")])],
            ["Distribution", ["distribution-panel"], "/analysis", true, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickClass("dash-cell column-0", 1, "selected-row")])],
            ["Distribution", ["replicate-and-norm", "distribution-panel"], "/analysis", true, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickClass("dash-cell column-0", 1, "selected-row")])],
            ["Distribution", ["distribution-panel"], "/analysis", true, null],
            ["Table", ["table-tut", "table-tab"], "/analysis", true, simulateClickID("table-tab", "table-tut")],
            ["Table", ["distribution-panel", "table-tut", "table-tab"], "/analysis", true, simulateClickID("table-tab", "table-tut")],
            ["Analysis Workflow", ["selector-box-tut"], "/analysis", false, null, true],
            ["Analysis Workflow", ["kernel-tut"], "/analysis", false, null, true],
            ["Analysis Workflow", ["distance-method-tut"], "/analysis", false, null, true],
            ["Analysis Workflow", ["heatmap-box-tut"], "/analysis", false, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickClass("dash-cell column-0", 1, "selected-row")])],
            ["Analysis Workflow", ["score-rank-tut"], "/analysis", false, null, true],
            ["Analysis Workflow", ["anosim-tut"], "/analysis", false, null, true],
            ["Analysis Workflow", ["table-tut", "table-tab", "selector-box-tut"], "/analysis", false, simulateClickID("score-btn", 1)],
            ["Analysis Workflow", ["table-tut", "table-tab", "selector-box-tut"], "/analysis", false, null],
            ["Analysis Workflow", ["table-tut", "table-tab", "selector-box-tut"], "/analysis", false, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickSorting("ANOSIM R"), simulateClickSorting("Mean Distance")])],
            ["Analysis Workflow", ["table-tut", "selector-box-tut","table-tab",  "distribution-graph"], "/analysis", false, simulateClickID("rank-btn", 1), true],
            ["Analysis Workflow", ["heatmap-box-tut"], "/analysis", false, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickClass("dash-cell column-0", 1)])],
            ["Analysis Workflow", ["distribution-panel"], "/analysis", false, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickClass("dash-cell column-0", 1)])],
            ["Analysis Workflow", ["export-tut"], "/analysis", true, null, true],
            ["Figure Creation", ["color-tut"], "/analysis", true, null, true],
            ["Figure Creation", ["table-tut", "table-tab"], "/analysis", true, multiClickInOne([simulateClickID("table-tab", "table-tut"), simulateClickTableInput( 3), simulateClickTableInput( 5)])],
            ["Figure Creation", null, "/figure_factory", false, simulateClickFormCheckInput(0)],
            ["Figure Creation", ["ff-tut-preview"], "/figure_factory", false, simulateClickID("ff-default", 1)],
            ["Figure Creation", ["ff-tut-preview"], "/figure_factory", false, multiClickInOne([simulateClickFormCheckInput(2), simulateClickID("ff-default", null) ])],
            ["Bubble Plot", null, "/analysis", true],
            ["Bubble Plot", ["distribution-panel", "dim-red-tab", "dim-red-tut"], "/analysis", true, simulateClickID("dim-red-tab", "dim-red-tut")],
            ["Bubble Plot", ["dim-red-tut", "distribution-panel", "dim-red-tab", ], "/analysis", true, simulateClickID("dim-red-tab", "dim-red-tut")],
            ["Bubble Plot", ["dim-red-tut", "distribution-panel", "dim-red-tab", ], "/analysis", true, simulateClickID("dim-red-tab", "dim-red-tut")],
            ["Finish",null, null, false, simulateClickID("dim-red-tab", "dim-red-tut")],

        ]

    }

});



document.addEventListener('keydown', (event) => {
    const currentInput = document.getElementsByClassName("dash-cell focused")[0];
    if (!currentInput) {
    // Break the code execution
    return ""
    // You may want to add any further actions here
    }
    const currentTr = currentInput.parentNode;
    switch (event.key) {
        case "ArrowUp":
            // Up pressed

            (currentTr.previousElementSibling.children[1]).focus();
            break;
        case "ArrowDown":
            // Down pressed
            (currentTr.nextElementSibling.children[1]).focus();
            break;
    }
})
//
document.addEventListener('click', (event) => {
    const clickedElement = event.target;

    // Check if the clicked element is a <div> inside a <td>
    if (clickedElement.tagName === 'DIV' && clickedElement.closest('td')) {
        const parent = clickedElement.parentNode;
        parent.focus();
        // Add your code to handle the click on the <div> inside the <td>
    } else if (clickedElement.tagName === 'INPUT' && clickedElement.closest('td')) {
        const parent = clickedElement.parentNode.nextElementSibling;
        document.activeElement.blur()
    }
})
//
//
//
// document.addEventListener('focus', function (event) {
//     const focusedElement = event.target;
//     if (focusedElement.tagName === 'TD') {
//         const elements = document.getElementsByClassName('selected-row'); // Replace with your class name
//         for (let i = 0; i < elements.length; i++) {
//             console.log(elements[i])
//             elements[i].classList.remove('selected-row');
//         }
//         const row = focusedElement.parentNode
//         row.classList.add("selected-row")
//
//         // Add your code to run when a table cell gains focus
//     }
//     // Add your code to run when an element gains focus
// }, true);

addEventListener("dragover", (event) => {
    const dragOverElement = event.target;
    if (dragOverElement.classList.contains("custom-tab")) {
        dragOverElement.click()
    }
});

var btn = document.getElementById("reset-rows-btn")
var container = document.getElementsByClassName("previous-next-container")[0];
container.insertBefore(btn, container.firstChild);



