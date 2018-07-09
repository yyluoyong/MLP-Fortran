module mod_BaseActivationFunction
implicit none
    
!-------------------
! 抽象类：激活函数 |
!-------------------
type, abstract, public :: BaseActivationFunction

!||||||||||||    
contains   !|
!||||||||||||

    !* 激活函数
    procedure(abs_f), deferred, public :: f 
    !* 接收向量参数的激活函数
    procedure(abs_f_vect), deferred, public :: f_vect 
    !* 激活函数导数
    procedure(abs_df), deferred, public :: df 
    !* 接收向量参数的激活函数导数
    procedure(abs_df_vect), deferred, public :: df_vect  

end type BaseActivationFunction
!===================
    

!-------------------
! 抽象类：函数接口 |
!-------------------	
abstract interface 

	!* 注：对于Sigmoid等激活函数，其是单变量函数，
	!* 而对于softmax函数来说，它的输入是一个向量，
	!* 是多变量函数，因此函数应接收向量 x.

	!* 激活函数
	subroutine abs_f( this, index, x, y )
    use mod_Precision
    import :: BaseActivationFunction
	implicit none
		class(BaseActivationFunction), intent(inout) :: this
		!* index 表示返回 f( x(index) ) 的值
		integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: y
		
		
	end subroutine
	!====
    
    !* 接收向量参数的激活函数
	subroutine abs_f_vect( this, x, y )
    use mod_Precision
    import :: BaseActivationFunction
	implicit none
		class(BaseActivationFunction), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: y
		
	end subroutine
	!====
	
	!* 激活函数一阶导数
	subroutine abs_df( this, index, x, dy  )
    use mod_Precision
    import :: BaseActivationFunction
	implicit none
		class(BaseActivationFunction), intent(inout) :: this
		!* index 表示返回 f'( x(index) ) 的值
		integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: dy

	end subroutine
	!====
	
	!* 接收向量参数的激活函数一阶导数
	subroutine abs_df_vect( this, x, dy )
    use mod_Precision
    import :: BaseActivationFunction
	implicit none
		class(BaseActivationFunction), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: dy

	end subroutine
	!====

end interface
!===================
    
end module