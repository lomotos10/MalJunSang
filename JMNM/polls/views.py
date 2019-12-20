from django.shortcuts import render
from .dothings.things import predict_pos_neg

#View for the initial page
def index(request):
    return render( request, 'index.html')

#View for the Redirecting page -- This is where I want to catch the text box input
def results(request):
    inp_value = request.GET.get('results', 'Please no XSS :( ')
    if inp_value == 'Please no XSS :( ':
        pass
    else:
        score = predict_pos_neg(inp_value)
        print(inp_value, score)
        if score > 0.7:
            inp_value = '😄'
        elif score > 0.3:
            inp_value = '🙂'
        else:
            inp_value = '💔'
    context = {'inp_value': inp_value}
    return render( request, 'polls/index.html', context)

'''
def show(request):
    inp_value = request.GET.get('text_search', 'This is a default value')
    inp_text = request.GET.get('results', 'This is a default text')
    inp_text = double(inp_text)
    context = {'inp_value': inp_value, 'inp_text': inp_text}
    return render(request, 'polls/name2.html', context)
    '''