# ImControl

ImControl is an immediate mode variable manipulation library for [Dear ImGui](https://github.com/ocornut/imgui).  It's API mimics the style of Dear ImGui, it's fast like Dear ImGui and it's easy to use, like Dear ImGui.
If you use variables to control, position and orient your objects in 1D, 2D or 3D and wish you could edit them by dragging the objects on screen then ImControl is for you.  You just state how to position some control points with your variables and let it do the rest.

What ImControl provides:

* a transformation stack to describe how you manipulate your objects with your variables
* a simple quadratic solver that feeds mouse-based interaction back to changes to your variables
* reparametrisation of variables (you may not realise it yet, but you want this...)
* a bunch of demos which would be a headache to implement without ImControl!
* example implementations of widgets, 3D gui elements, etc. built on top of ImControl.

What ImControl does not provide:

* a stable library of widgets, you can easily write your own though!
* rendering of anything (unless you ask it to), it is expected that you render your own objects, widgets etc.
* a physics simulation
* resolution of mathematical [singularities](https://en.wikipedia.org/wiki/Singularity_(mathematics)), you need to stay the right side of the [Inverse Function Theorem](https://en.wikipedia.org/wiki/Inverse_function_theorem)

![arm demo](https://user-images.githubusercontent.com/2971239/103889425-51301700-50de-11eb-8e90-e20fde6e0885.gif)

Read the source files for documentation.  To run the demos start with your chosen Dear ImGui implementation example, add a call to ImControl::NewFrame() after ImGui::NewFrame(), then call ImControl::ShowDemoWindow as you would ImGui's.

If you have any questions about the code then please contact me directly or raise an issue.  Furthermore if you have any suggestions for transforms / features you want included then again raise an issue.  Contributions to the code are of course welcome.

You may notice some strange behaviour when dragging certain control points around.  This may be an unavoidable result of your underlying geometry, the simplest way to mitigate such issues is to restrict the domain of the parameters, see the examples.

### Tests and Benchmarks

There are some simple tests I used when developing the transforms, they perform a numerical check of the implemented formal derivatives.  They are part of the demo.  There are also some benchmarks which compare the performance of the various types of transformations.
